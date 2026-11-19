"""WAV helpers built on the standard library only.

The service intentionally avoids a hard dependency on torch/soundfile for the
non-inference path so that role management keeps working even when the AuK
checkpoints or the GPU stack are unavailable.

Supported containers: RIFF/WAVE with PCM (format 1, 8/16/24/32-bit) and IEEE
float (format 3, 32/64-bit), any channel count.  Everything is normalised to
mono 16-bit PCM when a file is trimmed/stored, because that is what both AuK
stacks consume anyway.
"""

from __future__ import annotations

import io
import math
import re
import struct
import wave
from pathlib import Path

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_END_RE = re.compile(r"[。！？!?；;，,、.:：]")
DIGIT_GROUP_RE = re.compile(r"\d+(?:\.\d+)?")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[%％]")
# unit suffixes that must be read as Chinese words rather than Latin letters
UNIT_MAP = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs)\b", re.I), r"\1秒"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:min|mins)\b", re.I), r"\1分钟"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs)\b", re.I), r"\1小时"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg)\b", re.I), r"\1公斤"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:ml)\b", re.I), r"\1毫升"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*℃"), r"\1摄氏度"),
]

WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def _parse_riff(path: str | Path) -> dict:
    """Minimal RIFF/WAVE parser (fmt + data chunks), tolerant of extra chunks."""
    return _parse_riff_bytes(Path(path).read_bytes())


def _parse_riff_bytes(raw: bytes) -> dict:
    if len(raw) < 44 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")

    fmt: dict | None = None
    data_offset = data_size = None
    pos = 12
    while pos + 8 <= len(raw):
        chunk_id = raw[pos:pos + 4]
        chunk_size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = pos + 8
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError("truncated fmt chunk")
            tag, channels, rate, _byte_rate, _align, bits = struct.unpack("<HHIIHH", raw[body:body + 16])
            if tag == WAVE_FORMAT_EXTENSIBLE and chunk_size >= 40:
                tag = struct.unpack("<H", raw[body + 24:body + 26])[0]
            fmt = {"format": tag, "channels": channels, "sample_rate": rate, "bits": bits}
        elif chunk_id == b"data":
            data_offset, data_size = body, min(chunk_size, len(raw) - body)
            if fmt is not None and data_size is not None:
                # keep scanning only if fmt came after data (rare) — otherwise done
                pass
        pos = body + chunk_size + (chunk_size & 1)
        if fmt is not None and data_offset is not None:
            break

    if fmt is None or data_offset is None:
        raise ValueError("missing fmt/data chunk")
    if fmt["sample_rate"] <= 0 or fmt["channels"] <= 0 or fmt["bits"] <= 0:
        raise ValueError("invalid fmt chunk values")
    if fmt["format"] not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
        raise ValueError(f"unsupported WAVE format tag {fmt['format']} (PCM and IEEE float only)")

    frame_bytes = (fmt["bits"] // 8) * fmt["channels"]
    frames = data_size // frame_bytes if frame_bytes else 0
    return {
        **fmt,
        "data_offset": data_offset,
        "data_size": data_size,
        "frames": frames,
        "duration": round(frames / float(fmt["sample_rate"]), 3),
    }


def probe_wav(path: str | Path) -> dict:
    """Return format / sample rate / channels / duration of a WAV file."""
    info = _parse_riff(path)
    return {
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "sample_width": info["bits"] // 8,
        "frames": info["frames"],
        "duration": info["duration"],
        "format": "float" if info["format"] == WAVE_FORMAT_IEEE_FLOAT else "pcm",
        "bits": info["bits"],
    }


def is_riff_wave(path: str | Path) -> bool:
    try:
        _parse_riff(path)
        return True
    except Exception:
        return False


def probe_wav_bytes(data: bytes) -> dict:
    """Same as :func:`probe_wav` but for an in-memory WAV payload."""
    info = _parse_riff_bytes(data)
    return {
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "sample_width": info["bits"] // 8,
        "frames": info["frames"],
        "duration": info["duration"],
    }



def _to_mono_pcm16(frames: bytes, fmt: int, bits: int, channels: int) -> bytes:
    """Decode interleaved samples to mono int16 (channel average)."""
    width = bits // 8
    if fmt == WAVE_FORMAT_PCM and bits == 16 and channels == 1:
        return frames

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy ships with the AuK stack
        if fmt == WAVE_FORMAT_PCM and bits == 16:
            # downmix without numpy
            import array

            samples = array.array("h", frames)
            if channels == 1:
                return frames
            mono = array.array(
                "h",
                (
                    int(sum(samples[i:i + channels]) / channels)
                    for i in range(0, len(samples) - channels + 1, channels)
                ),
            )
            return mono.tobytes()
        raise ValueError("decoding non-PCM16 WAV requires numpy")

    if fmt == WAVE_FORMAT_PCM and bits == 8:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
        scaled = data * 256.0
    elif fmt == WAVE_FORMAT_PCM and bits == 16:
        scaled = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    elif fmt == WAVE_FORMAT_PCM and bits == 24:
        raw = np.frombuffer(frames[: len(frames) - len(frames) % 3], dtype=np.uint8)
        triples = raw.reshape(-1, 3).astype(np.int32)
        ints = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        scaled = (ints / 256.0).astype(np.float32)
    elif fmt == WAVE_FORMAT_PCM and bits == 32:
        scaled = (np.frombuffer(frames, dtype="<i4").astype(np.float64) / 65536.0).astype(np.float32)
    elif fmt == WAVE_FORMAT_IEEE_FLOAT and bits == 32:
        scaled = np.frombuffer(frames, dtype="<f4").astype(np.float32)
    elif fmt == WAVE_FORMAT_IEEE_FLOAT and bits == 64:
        scaled = np.frombuffer(frames, dtype="<f8").astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV layout: format={fmt} bits={bits}")

    if channels > 1 and scaled.size:
        usable = scaled.size - scaled.size % channels
        scaled = scaled[:usable].reshape(-1, channels).mean(axis=1)
    clipped = np.clip(scaled, -32768, 32767)
    return clipped.astype("<i2").tobytes()


def trim_wav(src: str | Path, dst: str | Path, max_seconds: float) -> dict:
    """Copy the first ``max_seconds`` of ``src`` into ``dst`` as mono PCM16."""
    info = _parse_riff(src)
    frame_bytes = (info["bits"] // 8) * info["channels"]
    limit = int(info["sample_rate"] * max_seconds) if max_seconds > 0 else info["frames"]
    keep = min(limit, info["frames"])
    raw = Path(src).read_bytes()
    chunk = raw[info["data_offset"]:info["data_offset"] + keep * frame_bytes]
    pcm = _to_mono_pcm16(chunk, info["format"], info["bits"], info["channels"])

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(pcm_wav_bytes(pcm, info["sample_rate"]))

    out = probe_wav(dst)
    out["trimmed"] = keep < info["frames"]
    out["source_duration"] = info["duration"]
    return out


def pcm_wav_bytes(frames: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)
    return buffer.getvalue()


def tone_wav_bytes(seconds: float, sample_rate: int = 24000, freq: float = 220.0,
                   amplitude: float = 0.05, channels: int = 1) -> bytes:
    """Placeholder audio for degraded mode: a quiet, faded sine tone.

    It is deliberately synthetic so that a caller can never mistake degraded
    output for AuK-generated speech.
    """
    total = max(1, int(seconds * sample_rate))
    fade = max(1, int(0.02 * sample_rate))
    frames = bytearray()
    for i in range(total):
        envelope = 1.0
        if i < fade:
            envelope = i / fade
        elif i > total - fade:
            envelope = max(0.0, (total - i) / fade)
        # two-tone so the output is unambiguous as a placeholder
        sample = amplitude * envelope * (
            math.sin(2 * math.pi * freq * i / sample_rate)
            + 0.5 * math.sin(2 * math.pi * freq * 1.5 * i / sample_rate)
        )
        value = int(max(-1.0, min(1.0, sample)) * 32767)
        packed = struct.pack("<h", value)
        frames.extend(packed * channels)
    return pcm_wav_bytes(bytes(frames), sample_rate, channels)


def normalize_text(text: str) -> str:
    """Prepare text for AuK: units become Chinese words, stray spaces around
    numbers are dropped so the model does not insert odd pauses.

    ``10.8s`` -> ``10.8秒``, ``90%`` -> ``百分之90``, ``优惠 30 %`` -> ``优惠百分之30``
    """
    text = (text or "").replace("\r", " ").strip()
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = PERCENT_RE.sub(r"百分之\1", text)
    for pattern, repl in UNIT_MAP:
        text = pattern.sub(repl, text)
    # remove spaces glued between Chinese characters and numbers
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=\d)", "", text)
    text = re.sub(r"(?<=\d)\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=\d)\s+(?=[%％])", "", text)
    return text.strip()


def spoken_seconds_of_numbers(text: str) -> float:
    """Approximate how long the digits in ``text`` take to read aloud."""
    total = 0.0
    for group in DIGIT_GROUP_RE.findall(text):
        # digits are read individually-ish: "199" -> 一百九十九, "10.8" -> 十点八
        syllables = max(2.0, len(group.replace(".", "")) * 1.3)
        if "." in group:
            syllables += 1.0
        total += syllables * 0.22
    total += text.count("百分之") * 0.66  # the "percent" prefix itself
    return total


def estimate_seconds(text: str, speed: float = 1.0) -> float:
    """Heuristic duration hint for zero-shot TTS.

    AuK needs an explicit output duration; without a transcript model the best
    available estimate is a language-aware character/word rate. Digits, units and
    percent signs must be counted too, otherwise numbers get squeezed.
    """
    text = (text or "").strip()
    if not text:
        return 1.0
    cjk = len(CJK_RE.findall(text))
    latin_words = len(WORD_RE.findall(CJK_RE.sub(" ", text)))
    pauses = len(SENTENCE_END_RE.findall(text))
    # ~4.6 Chinese chars/s, ~2.5 English words/s, ~120 ms per punctuation mark
    seconds = cjk / 4.6 + latin_words / 2.5 + pauses * 0.12 + spoken_seconds_of_numbers(text)
    speed = speed if speed and speed > 0 else 1.0
    return max(0.5, seconds / speed)


# --- long-text chunking ------------------------------------------------------
_SENT_END = "。！？!?…"
_CLAUSE_END = "，,、；;：:"


def split_text(text: str, *, max_seconds: float = 15.0, max_chars: int = 120,
               speed: float = 1.0) -> list[str]:
    """Split a script into pieces that each fit comfortably in one AuK call.

    Priority: paragraph / sentence boundary first, clause marks only when a single
    sentence is still too long, and a hard cut as the last resort. Every returned
    chunk keeps its trailing punctuation so the model sees natural phrasing.
    """
    text = (text or "").strip()
    if not text:
        return []

    def too_long(piece: str) -> bool:
        return estimate_seconds(piece, speed) > max_seconds or len(piece) > max_chars

    # 1) paragraphs, 2) sentences
    sentences: list[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        buffered = ""
        for piece in re.split(rf"(?<=[{re.escape(_SENT_END)}])", paragraph):
            if not piece:
                continue
            if buffered and too_long(buffered + piece):
                sentences.append(buffered)
                buffered = piece
            else:
                buffered += piece
        if buffered:
            sentences.append(buffered)

    # 3) greedy packing + clause/hard split for oversized sentences
    chunks: list[str] = []

    def push(piece: str) -> None:
        piece = piece.strip()
        if not piece:
            return
        if not too_long(piece):
            chunks.append(piece)
            return
        clauses = [c for c in re.split(rf"(?<=[{re.escape(_CLAUSE_END)}])", piece) if c]
        if len(clauses) > 1:
            buffered = ""
            for clause in clauses:
                if buffered and too_long(buffered + clause):
                    push(buffered)
                    buffered = clause
                else:
                    buffered += clause
            if buffered:
                push(buffered)
            return
        # hard cut as the last resort: prefer word boundaries for latin text
        cjk = len(CJK_RE.findall(piece))
        latin_chars = max(1, len(piece) - cjk)
        chars_per_second = 4.6 if cjk >= latin_chars else 13.0
        limit = max(20, int(max_seconds * chars_per_second * 0.9))
        remaining = piece
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break
            window = remaining[:limit]
            cut = max(window.rfind(" "), window.rfind("\u3000"))
            if cut < limit * 0.6:
                cut = limit
            chunks.append(remaining[:cut].rstrip())
            remaining = remaining[cut:].lstrip()

    buffered = ""
    for sentence in sentences:
        if buffered and too_long(buffered + sentence):
            push(buffered)
            buffered = sentence
        else:
            buffered += sentence
    if buffered:
        push(buffered)

    return [c for c in chunks if c.strip()]


def pause_after(chunk: str) -> float:
    """Natural silence to insert after a chunk, based on its trailing punctuation."""
    tail = chunk.rstrip()[-1:] if chunk else ""
    if tail in "。！？!?…":
        return 0.45
    if tail in "；;：:":
        return 0.3
    if tail in "，,、":
        return 0.18
    return 0.25


def join_audio(segments: list[tuple[bytes, float]], *, fade_ms: float = 15.0,
               match_rms: bool = True, max_gain_db: float = 6.0) -> tuple[bytes, int]:
    """Concatenate ``(wav_bytes, gap_after_seconds)`` pairs into one mono WAV."""
    import numpy as np

    rate: int | None = None
    clips: list[np.ndarray] = []
    for wav_bytes, _gap in segments:
        info = _parse_riff_bytes(wav_bytes)
        pcm = _to_mono_pcm16(wav_bytes[info["data_offset"]:info["data_offset"] + info["data_size"]],
                             info["format"], info["bits"], info["channels"])
        rate = rate or info["sample_rate"]
        clip = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        if info["sample_rate"] != rate:
            n = int(round(clip.size * rate / info["sample_rate"]))
            clip = np.interp(np.linspace(0, clip.size - 1, n), np.arange(clip.size), clip).astype(np.float32)
        clips.append(clip)

    if rate is None:
        return pcm_wav_bytes(b"", 24000), 24000

    # trim model-generated leading/trailing silence so the inserted pause is exact
    trimmed: list[np.ndarray] = []
    for clip in clips:
        if clip.size == 0:
            trimmed.append(clip)
            continue
        frame = max(1, int(rate * 0.01))
        frames = len(clip) // frame
        if frames < 3:
            trimmed.append(clip)
            continue
        energy = np.array([np.sqrt((clip[i * frame:(i + 1) * frame] ** 2).mean()) for i in range(frames)])
        peak = float(energy.max())
        if peak <= 0:
            trimmed.append(clip)
            continue
        loud = np.where(energy > peak * 0.03)[0]
        if loud.size == 0:
            trimmed.append(clip)
            continue
        margin = 2  # 20 ms of headroom on each side
        start = max(0, loud[0] - margin) * frame
        end = min(frames, loud[-1] + 1 + margin) * frame
        trimmed.append(clip[start:end])
    clips = trimmed

    if match_rms and len(clips) > 1:
        rms = np.array([float(np.sqrt((c ** 2).mean())) for c in clips])
        target = float(np.median(rms)) or 1.0
        cap = 10 ** (max_gain_db / 20)
        for i, clip in enumerate(clips):
            if rms[i] > 1e-6:
                clips[i] = clip * min(cap, max(1 / cap, target / rms[i]))

    fade = max(1, int(rate * fade_ms / 1000))
    pieces: list[np.ndarray] = []
    for index, clip in enumerate(clips):
        if clip.size > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            clip = clip.copy()
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        pieces.append(clip)
        gap = segments[index][1]
        if gap > 0 and index < len(clips) - 1:
            pieces.append(np.zeros(int(rate * gap), dtype=np.float32))

    merged = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    return pcm_wav_bytes(np.clip(merged, -32768, 32767).astype("<i2").tobytes(), rate), rate

def decode_mono(data_or_path) -> "tuple[Any, int]":
    """Decode a WAV (bytes or path) to a float32 numpy array + sample rate."""
    import numpy as np

    raw = data_or_path if isinstance(data_or_path, (bytes, bytearray)) else Path(data_or_path).read_bytes()
    info = _parse_riff_bytes(bytes(raw))
    pcm = _to_mono_pcm16(bytes(raw)[info["data_offset"]:info["data_offset"] + info["data_size"]],
                         info["format"], info["bits"], info["channels"])
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0, info["sample_rate"]


def similarity(a, b) -> float:
    """Waveform correlation in [-1, 1] between two WAVs (bytes or paths).

    Used to detect the failure mode where the resident server hands back the
    reference recording instead of synthesised speech.
    """
    import numpy as np

    xa, sra = decode_mono(a)
    xb, srb = decode_mono(b)
    if sra != srb and xb.size:  # compare on the shorter common duration after crude resample
        n = int(round(xb.size * sra / srb))
        xb = np.interp(np.linspace(0, xb.size - 1, n), np.arange(xb.size), xb).astype(np.float32)
    n = min(xa.size, xb.size)
    if n < 1000:
        return 0.0
    xa, xb = xa[:n] - xa[:n].mean(), xb[:n] - xb[:n].mean()
    denom = float(np.linalg.norm(xa) * np.linalg.norm(xb))
    return float(np.dot(xa, xb) / denom) if denom > 1e-9 else 0.0
