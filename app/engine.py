"""Inference backends for the speech service.

Two implementations are provided:

* ``AukEngine``  — thin wrapper around ``auk.infer.infer_auk.AukInfer`` (the
  official AuK pipeline: Qwen2.5-Omni text/audio encoder + CFM DiT + BigVGAN VAE).
* ``MockEngine`` — dependency-free placeholder that renders a short synthetic
  tone of the requested duration. Used only when the AuK stack or the
  checkpoints are unavailable, so that the HTTP contract stays testable.

The service never silently upgrades a placeholder to real speech: every
degraded response carries ``engine="mock"`` plus a ``warning`` field.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import audio as audio_utils
from . import audioserver
from . import config

logger = logging.getLogger("auk.service.engine")


ZERO_SHOT_TEMPLATE = 'Say the following with the same voice: "{text}"'
MAX_VOICE_DESC_CHARS = 80
INSTRUCT_TEMPLATE_ZH = '请基于下面的描述: "{desc}",生成语音内容"{text}".'
INSTRUCT_TEMPLATE_EN = 'Generate speech based on the following description: "{desc}". The content to speak is: "{text}".'


@dataclass
class SynthesisResult:
    wav_bytes: bytes
    sample_rate: int
    duration: float
    engine: str
    mode: str
    gen_seconds: float
    notes: list[str] = field(default_factory=list)      # informational, request succeeded
    warnings: list[str] = field(default_factory=list)   # something needs attention
    elapsed: float = 0.0


class EngineUnavailable(RuntimeError):
    """Raised when the requested backend cannot serve a request."""


# --------------------------------------------------------------------------- mock
class MockEngine:
    name = "mock"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def status(self) -> dict:
        return {
            "active": "mock",
            "degraded": True,
            "reason": self.reason,
            "model_loaded": False,
        }

    def synthesize(self, text: str, *, ref_audio: Path | None, voice_description: str | None,
                   gen_seconds: float | None, speed: float, seed: int | None,
                   nfe: int, cfg_strength: float,
                   max_chunk_seconds: float | None = None,
                   role_name: str | None = None) -> SynthesisResult:
        started = time.perf_counter()
        seconds = gen_seconds or audio_utils.estimate_seconds(text, speed)
        seconds = min(seconds, config.MAX_GEN_SECONDS)
        wav = audio_utils.tone_wav_bytes(seconds, config.TARGET_SAMPLE_RATE)
        warnings = [f"降级模式：{self.reason}"]
        return SynthesisResult(
            wav_bytes=wav,
            sample_rate=config.TARGET_SAMPLE_RATE,
            duration=round(seconds, 3),
            engine=self.name,
            mode="zero_shot_tts" if ref_audio else "instruct_tts",
            gen_seconds=round(seconds, 3),
            warnings=warnings,
            elapsed=round(time.perf_counter() - started, 3),
        )

    def edit(self, instruction: str, source_audio: Path, *,
             duration_seconds: float | None = None, seed: int | None = None) -> SynthesisResult:
        started = time.perf_counter()
        info = audio_utils.probe_wav(source_audio)
        seconds = min(duration_seconds or info["duration"], config.MAX_GEN_SECONDS)
        wav = audio_utils.tone_wav_bytes(seconds, config.TARGET_SAMPLE_RATE)
        return SynthesisResult(
            wav_bytes=wav,
            sample_rate=config.TARGET_SAMPLE_RATE,
            duration=round(seconds, 3),
            engine=self.name,
            mode="edit",
            gen_seconds=round(seconds, 3),
            warnings=[f"degraded mode: {self.reason}"],
            elapsed=round(time.perf_counter() - started, 3),
        )


# ---------------------------------------------------------------------------- auk
class AukEngine:
    name = "auk"

    def __init__(self, ckpt: Path, cfg: Path, qwen: Path | None, *, device: str = "cuda",
                 dtype: str = "bf16", cpu_offload: bool = False) -> None:
        self.ckpt = Path(ckpt)
        self.cfg = Path(cfg)
        self.qwen = Path(qwen) if qwen else None
        self.device = device
        self.dtype = dtype
        self.cpu_offload = cpu_offload
        self._infer = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        with self._load_lock:
            if self._infer is not None:
                return
            if self._load_error:
                raise EngineUnavailable(self._load_error)

            src = str(config.AUK_SRC)
            if src not in sys.path:
                sys.path.insert(0, src)
            try:
                from auk.infer.infer_auk import AukInfer  # noqa: WPS433 (runtime import)
            except Exception as exc:  # torch / transformers missing
                self._load_error = f"cannot import auk inference stack: {exc}"
                raise EngineUnavailable(self._load_error) from exc

            logger.info("loading AuK checkpoint %s (device=%s dtype=%s offload=%s)",
                        self.ckpt, self.device, self.dtype, self.cpu_offload)
            try:
                self._infer = AukInfer(
                    str(self.cfg),
                    str(self.ckpt),
                    device=self.device,
                    dtype=self.dtype,
                    qwen_path=str(self.qwen) if self.qwen else None,
                    cpu_offload=self.cpu_offload,
                )
            except Exception as exc:
                self._load_error = f"AuK model load failed: {type(exc).__name__}: {exc}"
                raise EngineUnavailable(self._load_error) from exc
            logger.info("AuK model ready on %s", self.device)

    def status(self) -> dict:
        return {
            "active": "auk",
            "degraded": False,
            "reason": self._load_error,
            "model_loaded": self._infer is not None,
            "checkpoint": str(self.ckpt),
            "config": str(self.cfg),
            "text_encoder": str(self.qwen) if self.qwen else None,
            "device": self.device,
            "dtype": self.dtype,
            "cpu_offload": self.cpu_offload,
        }

    @property
    def sample_rate(self) -> int:
        return int(getattr(self._infer, "target_sample_rate", config.TARGET_SAMPLE_RATE))

    # --------------------------------------------------------------- generation
    def synthesize(self, text: str, *, ref_audio: Path | None, voice_description: str | None,
                   gen_seconds: float | None, speed: float, seed: int | None,
                   nfe: int, cfg_strength: float,
                   max_chunk_seconds: float | None = None,
                   role_name: str | None = None) -> SynthesisResult:
        self.load()
        started = time.perf_counter()

        if ref_audio is not None:
            instruction = ZERO_SHOT_TEMPLATE.format(text=text)
            content = [
                {"type": "text", "text": instruction},
                {"type": "audio", "audio": str(ref_audio)},
            ]
            mode = "zero_shot_tts"
        else:
            description = voice_description or "一位成年说话人，语调自然，吐字清晰。"
            has_cjk = audio_utils.CJK_RE.search(text) or audio_utils.CJK_RE.search(description)
            template = INSTRUCT_TEMPLATE_ZH if has_cjk else INSTRUCT_TEMPLATE_EN
            content = [{"type": "text", "text": template.format(desc=description, text=text)}]
            mode = "instruct_tts"

        seconds = gen_seconds or audio_utils.estimate_seconds(text, speed)
        seconds = min(seconds, config.MAX_GEN_SECONDS)

        warnings: list[str] = []
        if ref_audio is not None:
            ref_seconds = audio_utils.probe_wav(ref_audio)["duration"]
            if ref_seconds + seconds > 30:
                warnings.append(
                    f"reference {ref_seconds:.1f}s + target {seconds:.1f}s exceeds the ~30s "
                    "source+target budget; expect lower fidelity"
                )

        with self._generate_lock:  # single GPU: serialise generation
            audio_out, sr = self._infer.generate(
                [{"role": "user", "content": content}],
                gen_seconds=seconds,
                nfe=nfe,
                cfg_strength=cfg_strength,
                seed=seed,
            )

        tmp_path = config.TMP_DIR / f"auk_{uuid.uuid4().hex}.wav"
        try:
            from auk.infer.infer_auk import save_audio

            save_audio(audio_out, sr, str(tmp_path))
            wav_bytes = tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

        duration = round(float(audio_out.shape[-1]) / sr, 3) if hasattr(audio_out, "shape") else round(seconds, 3)

        return SynthesisResult(
            wav_bytes=wav_bytes,
            sample_rate=int(sr),
            duration=duration,
            engine=self.name,
            mode=mode,
            gen_seconds=round(seconds, 3),
            warnings=warnings,
            elapsed=round(time.perf_counter() - started, 3),
        )

    def edit(self, instruction: str, source_audio: Path, *,
             duration_seconds: float | None = None, seed: int | None = None) -> SynthesisResult:
        """Python-stack equivalent of the GGUF edit route (instruction + source audio)."""
        self.load()
        started = time.perf_counter()
        seconds = duration_seconds or audio_utils.probe_wav(source_audio)["duration"]
        seconds = min(seconds, config.MAX_GEN_SECONDS)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "audio", "audio": str(source_audio)},
                ],
            }
        ]
        with self._generate_lock:
            audio_out, sr = self._infer.generate(messages, gen_seconds=seconds, seed=seed)

        tmp_path = config.TMP_DIR / f"auk_{uuid.uuid4().hex}.wav"
        try:
            from auk.infer.infer_auk import save_audio

            save_audio(audio_out, sr, str(tmp_path))
            wav_bytes = tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

        duration = round(float(audio_out.shape[-1]) / sr, 3) if hasattr(audio_out, "shape") else round(seconds, 3)
        return SynthesisResult(
            wav_bytes=wav_bytes,
            sample_rate=int(sr),
            duration=duration,
            engine=self.name,
            mode="edit",
            gen_seconds=round(seconds, 3),
            elapsed=round(time.perf_counter() - started, 3),
        )


# --------------------------------------------------------------------------- gguf
class AukGgufEngine:
    """AuK via audio.cpp (GGUF/ggml) — no Python torch stack involved.

    The CLI is spawned per request: ``audiocpp_cli --task tts --family auk``.
    Model weights are quantised GGUFs (q8_0 by default), which keeps the whole
    pipeline inside ~7 GB of VRAM instead of the ~17-25 GB the Python stack needs.
    """

    name = "gguf"

    def __init__(self, cli: Path, model_dir: Path, *, variant: str = "flash",
                 backend: str = "cuda", mem_saver: bool = False,
                 variant_files: dict | None = None,
                 edit_variant: str = "base") -> None:
        self.cli = Path(cli)
        self.model_dir = Path(model_dir)
        self.variant = variant
        self.edit_variant = edit_variant
        self.backend = backend
        self.mem_saver = mem_saver
        self.variant_files = variant_files or {}
        self._lock = threading.Lock()
        self._cli_version: str | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def components(self) -> dict:
        return self.variant_files.get(self.variant, {})

    def _env(self) -> dict:
        env = os.environ.copy()
        cuda_bin = config.cuda_bin_dir()
        if cuda_bin is not None:
            env["PATH"] = f"{cuda_bin}{os.pathsep}{env.get('PATH', '')}"
        return env

    def status(self) -> dict:
        files = self.components
        generator = self.model_dir / files.get("generator", "")
        return {
            "active": "gguf",
            "degraded": False,
            "reason": None,
            "model_loaded": False,  # stateless CLI: nothing stays resident
            "cli": str(self.cli),
            "cli_present": self.cli.is_file(),
            "model_dir": str(self.model_dir),
            "variant": self.variant,
            "backend": self.backend,
            "edit_variant": self.edit_variant,
            "edit_variant_present": (
                self.model_dir / (self.variant_files.get(self.edit_variant) or {}).get("generator", "-")
            ).is_file(),
            "components": {
                "generator": str(generator),
                "generator_present": generator.is_file(),
                "qwen": str(self.model_dir / config.GGUF_QWEN),
                "vae": str(self.model_dir / config.GGUF_VAE),
            },
            "cuda_bin": str(config.cuda_bin_dir()) if config.cuda_bin_dir() else None,
        }

    # --------------------------------------------------------------- generation
    def _base_cmd(self, task: str, variant: str | None = None) -> list[str]:
        variant = variant or self.variant
        files = self.variant_files.get(variant) or {}
        if not files:
            raise EngineUnavailable(f"unknown AuK GGUF variant: {variant}")
        if not self.cli.is_file():
            raise EngineUnavailable(f"audio.cpp CLI not found: {self.cli}")
        generator = self.model_dir / files["generator"]
        if not generator.is_file():
            raise EngineUnavailable(
                f"GGUF generator for variant '{variant}' is missing: {generator.name} "
                f"(download it into {self.model_dir})"
            )
        cmd = [
            str(self.cli),
            "--task", task,
            "--family", "auk",
            "--model", str(self.model_dir),
            "--backend", self.backend,
            "--session-option", f"auk.variant={variant}",
            "--session-option", f"auk.model_gguf={files['generator']}",
            "--session-option", f"auk.qwen_gguf={config.GGUF_QWEN}",
            "--session-option", f"auk.vae_gguf={config.GGUF_VAE}",
        ]
        if self.mem_saver:
            cmd += ["--session-option", "auk.mem_saver=true"]
        return cmd

    def _execute(self, cmd: list[str], out_path: Path, warnings: list[str],
                 cleanup: list[Path] | None = None, keep_output: bool = False) -> tuple[bytes, dict]:
        logger.debug("audiocpp: %s", " ".join(cmd))
        try:
            with self._lock:  # single GPU: serialise CLI invocations
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.cli.parent),
                    env=self._env(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=config.GGUF_TIMEOUT,
                )
        except subprocess.TimeoutExpired as exc:
            raise EngineUnavailable(
                f"audiocpp_cli timed out after {config.GGUF_TIMEOUT:.0f}s"
            ) from exc
        finally:
            for path in cleanup or []:
                Path(path).unlink(missing_ok=True)

        if proc.returncode != 0:
            # audio.cpp 0.9.5 (Windows) can raise an access violation while tearing
            # down the graph *after* the WAV has been written. Judge the request by
            # the artifact, not by the exit code alone.
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            if out_path.is_file() and audio_utils.probe_wav(out_path)["frames"] > 0:
                warnings.append(
                    f"audiocpp_cli 退出码 {proc.returncode}（音频写出后进程崩溃），音频完整"
                )
                logger.warning("audiocpp_cli exit=%s but output is valid, continuing", proc.returncode)
            else:
                raise RuntimeError(
                    f"audiocpp_cli failed (exit {proc.returncode}): {' / '.join(tail)}"
                )

        try:
            wav_bytes = out_path.read_bytes()
            info = audio_utils.probe_wav(out_path)
        finally:
            if not keep_output:
                out_path.unlink(missing_ok=True)
        return wav_bytes, info

    def synthesize(self, text: str, *, ref_audio: Path | None, voice_description: str | None,
                   gen_seconds: float | None, speed: float, seed: int | None,
                   nfe: int, cfg_strength: float,
                   max_chunk_seconds: float | None = None,
                   role_name: str | None = None) -> SynthesisResult:
        started = time.perf_counter()
        notes: list[str] = []
        warnings: list[str] = []
        cleanup: list[Path] = []

        text = audio_utils.normalize_text(text)
        if voice_description:
            voice_description, truncated = clean_description(voice_description)
            if truncated:
                notes.append(f"音色描述超过 {MAX_VOICE_DESC_CHARS} 字，已截断以避免被朗读")

        # Long scripts must not be squeezed into the 20 s ceiling: split them so each
        # chunk gets its own duration budget, then join with punctuation-aware pauses.
        limit = config.MAX_CHUNK_SECONDS if max_chunk_seconds is None else max_chunk_seconds
        estimated = gen_seconds or audio_utils.estimate_seconds(text, speed)
        if gen_seconds is None and limit and estimated > limit and len(text.strip()) > 40:
            return self._synthesize_chunked(
                text, ref_audio=ref_audio, voice_description=voice_description, speed=speed,
                seed=seed, nfe=nfe, cfg_strength=cfg_strength, limit=limit, started=started,
            )

        cmd = self._base_cmd("tts")

        seconds = gen_seconds or audio_utils.estimate_seconds(text, speed)
        seconds = min(seconds, config.MAX_GEN_SECONDS)

        if ref_audio is not None:
            # zero-shot: the text must carry the complete upstream instruction.
            # Long references (up to 27 s in ./role) slow AuK down and eat into the
            # ~30 s source+target budget, so clamp them to REF_TRIM_SECONDS.
            ref_seconds = audio_utils.probe_wav(ref_audio)["duration"]
            ref_path = ref_audio
            if ref_seconds > config.REF_TRIM_SECONDS:
                ref_path = config.TMP_DIR / f"ref_{uuid.uuid4().hex}.wav"
                audio_utils.trim_wav(ref_audio, ref_path, config.REF_TRIM_SECONDS)
                cleanup.append(ref_path)
                notes.append(
                    f"参考音频 {ref_seconds:.1f} 秒，已截取前 {config.REF_TRIM_SECONDS:.0f} 秒作为音色条件"
                )
                ref_seconds = config.REF_TRIM_SECONDS
            cmd += ["--voice-ref", str(ref_path)]
            cmd += ["--text", ZERO_SHOT_TEMPLATE.format(text=text)]
            mode = "zero_shot_tts"
            if ref_seconds + seconds > 30:
                warnings.append(
                    f"参考 {ref_seconds:.1f} 秒 + 目标 {seconds:.1f} 秒超过约 30 秒预算，音质可能下降"
                )
        else:
            description = voice_description or "一位成年说话人，语调自然，吐字清晰。"
            cmd += ["--text", text]
            cmd += ["--request-option", f"instruct={description}"]
            mode = "instruct_tts"

        cmd += ["--request-option", f"duration_sec={seconds:.2f}"]
        if self.variant == "base":
            cmd += ["--request-option", f"num_inference_steps={nfe}"]
            cmd += ["--request-option", f"guidance_scale={cfg_strength}"]
        cmd += ["--seed", str(seed if seed is not None else -1)]

        out_path = config.TMP_DIR / f"gguf_{uuid.uuid4().hex}.wav"
        cmd += ["--out", str(out_path)]
        logger.info("audiocpp tts: variant=%s backend=%s dur=%.2fs ref=%s",
                    self.variant, self.backend, seconds, ref_audio.name if ref_audio else None)

        wav_bytes, info = self._execute(cmd, out_path, warnings, cleanup)

        if mode == "instruct_tts" and info["duration"] > max(seconds * 2.2, seconds + 2.0):
            warnings.append(
                f"输出 {info['duration']:.1f} 秒明显超过文本所需的 {seconds:.1f} 秒，"
                "可能把音色描述也朗读了；建议缩短描述或改用角色参考音频"
            )

        return SynthesisResult(
            wav_bytes=wav_bytes,
            sample_rate=int(info["sample_rate"]),
            duration=float(info["duration"]),
            engine=self.name,
            mode=mode,
            gen_seconds=round(seconds, 3),
            notes=notes,
            warnings=warnings,
            elapsed=round(time.perf_counter() - started, 3),
        )

    def _synthesize_chunked(self, text: str, *, ref_audio: Path | None,
                            voice_description: str | None, speed: float, seed: int | None,
                            nfe: int, cfg_strength: float, limit: float,
                            started: float) -> SynthesisResult:
        """Split a long script, synthesize each chunk with its own duration, join."""
        chunks = audio_utils.split_text(text, max_seconds=limit, speed=speed)
        if len(chunks) <= 1:
            # nothing gained from splitting: fall back to a single call
            return self.synthesize(
                text, ref_audio=ref_audio, voice_description=voice_description,
                gen_seconds=min(audio_utils.estimate_seconds(text, speed), config.MAX_GEN_SECONDS),
                speed=speed, seed=seed, nfe=nfe, cfg_strength=cfg_strength, max_chunk_seconds=0,
            )

        segments: list[tuple[bytes, float]] = []
        notes: list[str] = []
        warnings: list[str] = []
        for index, chunk in enumerate(chunks):
            seconds = min(audio_utils.estimate_seconds(chunk, speed), config.MAX_GEN_SECONDS)
            one = self.synthesize(
                chunk, ref_audio=ref_audio, voice_description=voice_description,
                gen_seconds=seconds, speed=speed, seed=(seed or 0) + index,
                nfe=nfe, cfg_strength=cfg_strength, max_chunk_seconds=0,
            )
            notes.extend(n for n in one.notes if n not in notes)
            warnings.extend(w for w in one.warnings if w not in warnings)
            segments.append((one.wav_bytes, audio_utils.pause_after(chunk) * config.CHUNK_GAP_SCALE))

        wav_bytes, rate = audio_utils.join_audio(segments)
        info = audio_utils.probe_wav_bytes(wav_bytes)
        mode = "instruct_tts" if ref_audio is None else "zero_shot_tts"
        notes.append(
            f"长文本已切分为 {len(chunks)} 段（每段 ≤ {limit:.0f} 秒）以保证语速自然"
        )
        logger.info("chunked synthesis: %d chunks, out=%.2fs", len(chunks), info["duration"])
        return SynthesisResult(
            wav_bytes=wav_bytes,
            sample_rate=int(info["sample_rate"]),
            duration=float(info["duration"]),
            engine=self.name,
            mode=f"{mode}+chunked",
            gen_seconds=round(float(info["duration"]), 3),
            notes=notes,
            warnings=warnings,
            elapsed=round(time.perf_counter() - started, 3),
        )

    # ------------------------------------------------------------------- editing
    def edit(self, instruction: str, source_audio: Path, *,
             duration_seconds: float | None = None, seed: int | None = None,
             variant: str | None = None) -> SynthesisResult:
        """Instruction-driven editing / enhancement / separation on an existing clip.

        Covers every task in the README table except the two speech-generation ones:
        content & lyric editing, pitch/speed/volume, emotion/timbre/de-accent/nonverbal/
        whisper, speech enhancement, speech & music separation, target speaker extraction.
        """
        started = time.perf_counter()
        info_in = audio_utils.probe_wav(source_audio)
        warnings: list[str] = []
        notes: list[str] = []

        if info_in["duration"] > 30:
            warnings.append(
                f"source {info_in['duration']:.1f}s exceeds the ~30s single-clip budget; "
                "the tail may be truncated"
            )

        # editing is much more faithful on the base (32-step, CFG) schedule than on
        # the distilled Flash one; fall back to the generation variant when the base
        # generator is not downloaded
        use_variant = variant or self.edit_variant
        if use_variant != self.variant and not (
            self.model_dir / (self.variant_files.get(use_variant) or {}).get("generator", "-")
        ).is_file():
            notes.append(f"未安装 {use_variant} 模型，本次使用 {self.variant}（编辑效果较弱）")
            use_variant = self.variant

        cmd = self._base_cmd("gen", use_variant)
        cmd += ["--audio", str(source_audio), "--text", instruction]
        if duration_seconds:
            cmd += ["--request-option", f"duration_sec={duration_seconds:.2f}"]
        cmd += ["--seed", str(seed if seed is not None else -1)]

        out_path = config.TMP_DIR / f"gguf_{uuid.uuid4().hex}.wav"
        cmd += ["--out", str(out_path)]
        logger.info("audiocpp gen: variant=%s backend=%s src=%.2fs", use_variant,
                    self.backend, info_in["duration"])

        wav_bytes, info = self._execute(cmd, out_path, warnings)
        return SynthesisResult(
            wav_bytes=wav_bytes,
            sample_rate=int(info["sample_rate"]),
            duration=float(info["duration"]),
            engine=self.name,
            mode="edit",
            gen_seconds=round(duration_seconds or info["duration"], 3),
            notes=notes,
            warnings=warnings,
            elapsed=round(time.perf_counter() - started, 3),
        )



# --------------------------------------------------------------------- server
class AukServerEngine:
    """AuK through the resident ``audiocpp_server.exe`` process.

    Weights stay in VRAM, so a 3 s utterance costs ~0.3 s instead of ~7 s. The CLI
    engine is kept for editing (``/edit``), which the server does not expose for
    uploaded audio.
    """

    name = "gguf-server"

    def __init__(self, *, url: str, model_id: str, voice_dir: Path,
                 cli_engine: "AukGgufEngine | None" = None,
                 server_process=None, server_spawned: bool = False) -> None:
        self.url = url.rstrip("/")
        self.model_id = model_id
        self.voice_dir = Path(voice_dir)
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        self.cli_engine = cli_engine
        self.server_process = server_process
        self.server_spawned = server_spawned
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        info = audioserver.models(self.url)
        loaded = next((m for m in info if m.get("id") == self.model_id), None)
        return {
            "active": self.name,
            "degraded": loaded is None,
            "reason": None if loaded else "常驻服务未就绪",
            "model_loaded": bool(loaded and loaded.get("loaded")),
            "server": self.url,
            "model": self.model_id,
            "variant": config.GGUF_VARIANT,
            "backend": config.GGUF_BACKEND,
            "voice_dir": str(self.voice_dir),
            "soxr_dll": (config.AUDIO_CPP_DIR / "libsoxr.dll").is_file(),
            "gomp_dll": (config.AUDIO_CPP_DIR / "libgomp-1.dll").is_file(),
            "winpthread_dll": (config.AUDIO_CPP_DIR / "libwinpthread-1.dll").is_file(),
            "soxr_loadable": soxr_chain_report()["loadable"],
            "edit_backend": "cli" if self.cli_engine is not None else None,
        }

    # ------------------------------------------------------------------ voices
    def prepare_voice(self, role_path: Path, role_name: str) -> str:
        """Mirror a role's reference audio into the server's voice library."""
        safe = re.sub(r'[\\/:*?"<>|]', "_", role_name).strip() or "voice"
        target = self.voice_dir / f"{safe}.wav"
        source = Path(role_path)
        with self._lock:
            if not target.is_file() or source.stat().st_mtime > target.stat().st_mtime:
                # normalise to mono PCM16 and clamp length (shorter ref = faster encode)
                audio_utils.trim_wav(source, target, config.REF_TRIM_SECONDS)
                logger.info("音色库同步：%s -> %s", role_name, target.name)
        return safe

    def forget_voice(self, role_name: str) -> None:
        """Drop a deleted role's reference file from the server voice library."""
        safe = re.sub(r'[\\/:*?"<>|]', "_", role_name).strip() or "voice"
        target = self.voice_dir / f"{safe}.wav"
        target.unlink(missing_ok=True)

    # -------------------------------------------------------------- recovery
    def _recover(self, reason: str) -> None:
        """Restart the resident server (only when this service started it)."""
        if not self.server_spawned:
            raise EngineUnavailable(
                f"{reason}；当前常驻实例不是本服务启动的，请手动重启 audiocpp_server"
            )
        logger.warning("检测到 %s，正在重启常驻服务 …", reason)
        audioserver.stop(self.server_process)
        proc, spawned = audioserver.ensure(self.url)
        self.server_process, self.server_spawned = (proc or self.server_process), spawned

    # -------------------------------------------------------------- generation
    _RETRY_HINTS = ("soxr", "resampling")

    def _request(self, text: str, *, voice: str | None, seconds: float,
                 instruct: str | None, seed: int | None,
                 attempts: int = 3) -> tuple[bytes, dict]:
        options: dict = {"duration_sec": round(float(seconds), 2)}
        if instruct:
            options["instruct"] = instruct
        # With a reference voice AuK must receive the full instruction, exactly like the
        # CLI path (--voice-ref + "Say the following with the same voice: ..."). The
        # server forwards ``input`` verbatim, so a bare sentence makes the model treat it
        # as an instruction instead of target text — measured: correlation 0.09 against
        # the CLI baseline for bare text vs 1.000 for the wrapped form.
        body_text = ZERO_SHOT_TEMPLATE.format(text=text) if voice else text
        payload: dict = {"model": self.model_id, "input": body_text, "options": options}
        if voice:
            payload["voice"] = voice
        if seed is not None:
            payload["seed"] = int(seed)

        last_detail = ""
        for attempt in range(1, attempts + 1):
            try:
                status, body = audioserver.request_json(
                    f"{self.url}/v1/audio/speech", payload, timeout=config.GGUF_TIMEOUT
                )
            except ConnectionError as exc:
                # the resident process can die (OOM, killed, crash); bring it back
                if attempt < attempts and config.SERVER_AUTOSTART:
                    logger.warning("常驻服务连接失败，尝试自动拉起：%s", exc)
                    try:
                        proc, spawned = audioserver.ensure(self.url)
                        if spawned:
                            self.server_process, self.server_spawned = proc, True
                        continue
                    except Exception as restart_exc:  # noqa: BLE001
                        raise EngineUnavailable(
                            f"常驻服务不可用，重新拉起失败：{restart_exc}"
                        ) from restart_exc
                raise EngineUnavailable(f"常驻语音服务不可用：{exc}") from exc
            if status == 200 and isinstance(body, bytes):
                return body, audio_utils.probe_wav_bytes(body)

            detail = body.get("error", {}).get("message") if isinstance(body, dict) else str(body)[:300]
            last_detail = str(detail)
            # libsoxr is loaded lazily on the first reference-audio request; on Windows
            # that first LoadLibrary can fail transiently (AV scan / file in use), which
            # surfaces as a permanent-looking "requires Soxr HQ resampling" error.
            transient = status >= 500 and any(h in last_detail.lower() for h in self._RETRY_HINTS)
            if transient and attempt < attempts:
                logger.warning("常驻服务首次加载 soxr 失败（第 %d/%d 次），0.8s 后重试",
                               attempt, attempts)
                time.sleep(0.8)
                continue
            if transient:
                raise EngineUnavailable(
                    "常驻服务的音频重采样库（libsoxr）初始化失败，已重试 "
                    f"{attempts} 次仍失败；请重启常驻服务（或确认 {config.AUDIO_CPP_DIR} 下 "
                    "libsoxr.dll 与 libgomp-1.dll 存在且未被占用）"
                )
            raise RuntimeError(f"常驻服务合成失败（HTTP {status}）：{detail}")

        raise RuntimeError(f"常驻服务合成失败：{last_detail}")

    def synthesize(self, text: str, *, ref_audio: Path | None, voice_description: str | None,
                   gen_seconds: float | None, speed: float, seed: int | None,
                   nfe: int, cfg_strength: float,
                   max_chunk_seconds: float | None = None,
                   role_name: str | None = None) -> SynthesisResult:
        started = time.perf_counter()
        text = audio_utils.normalize_text(text)
        notes: list[str] = []
        warnings: list[str] = []

        instruct = None
        if voice_description and ref_audio is None:
            instruct, truncated = clean_description(voice_description)
            if truncated:
                notes.append(f"音色描述超过 {MAX_VOICE_DESC_CHARS} 字，已截断以避免被朗读")

        limit = config.MAX_CHUNK_SECONDS if max_chunk_seconds is None else max_chunk_seconds
        estimated = gen_seconds or audio_utils.estimate_seconds(text, speed)
        if gen_seconds is None and limit and estimated > limit and len(text.strip()) > 40:
            return self._synthesize_chunked(
                text, ref_audio=ref_audio, voice_description=voice_description, speed=speed,
                seed=seed, limit=limit, started=started, notes=notes, warnings=warnings,
                role_name=role_name,
            )

        seconds = min(estimated, config.MAX_GEN_SECONDS)
        voice = self.prepare_voice(ref_audio, role_name or ref_audio.stem) if ref_audio else None
        wav_bytes, info = self._request(text, voice=voice, seconds=seconds,
                                        instruct=instruct, seed=seed)
        if voice:
            wav_bytes, info = self._guard_echo(wav_bytes, info, voice, text,
                                               seconds=seconds, instruct=instruct, seed=seed)

        if instruct and info["duration"] > max(seconds * 2.2, seconds + 2.0):
            warnings.append(
                f"输出 {info['duration']:.1f} 秒明显超过文本所需的 {seconds:.1f} 秒，"
                "可能把音色描述也朗读了；建议缩短描述或改用角色参考音频"
            )

        return SynthesisResult(
            wav_bytes=wav_bytes,
            sample_rate=int(info["sample_rate"]),
            duration=float(info["duration"]),
            engine=self.name,
            mode="zero_shot_tts" if voice else "instruct_tts",
            gen_seconds=round(seconds, 3),
            notes=notes,
            warnings=warnings,
            elapsed=round(time.perf_counter() - started, 3),
        )

    def _guard_echo(self, wav_bytes: bytes, info: dict, voice: str, text: str, *,
                    seconds: float, instruct: str | None,
                    seed: int | None) -> tuple[bytes, dict]:
        """Catch the silent-failure mode where the reference audio is returned.

        When libsoxr failed to initialise, the resident server answers 200 with the
        prompt recording instead of synthesised speech. Detect it and recover once.
        """
        reference = self.voice_dir / f"{voice}.wav"
        if not reference.is_file():
            return wav_bytes, info
        try:
            score = audio_utils.similarity(wav_bytes, reference)
        except Exception:  # noqa: BLE001 - never fail a request on diagnostics
            return wav_bytes, info
        if score < 0.85:
            return wav_bytes, info

        logger.warning("输出与参考音频高度相似（%.2f），判定为重采样库未初始化", score)
        self._recover(f"常驻服务返回了参考音频（相似度 {score:.2f}），重采样库未初始化")
        wav_bytes, info = self._request(text, voice=voice, seconds=seconds,
                                        instruct=instruct, seed=seed)
        score = audio_utils.similarity(wav_bytes, reference)
        if score >= 0.85:
            raise EngineUnavailable(
                f"常驻服务仍然返回参考音频（相似度 {score:.2f}），音频重采样库不可用；"
                "请重启常驻服务并确认 libsoxr.dll / libgomp-1.dll 可用"
            )
        return wav_bytes, info

    def _synthesize_chunked(self, text: str, *, ref_audio: Path | None,
                            voice_description: str | None, speed: float, seed: int | None,
                            limit: float, started: float, notes: list[str],
                            warnings: list[str], role_name: str | None) -> SynthesisResult:
        chunks = audio_utils.split_text(text, max_seconds=limit, speed=speed)
        voice = self.prepare_voice(ref_audio, role_name or ref_audio.stem) if ref_audio else None
        instruct = None
        if voice_description and ref_audio is None:
            instruct, _ = clean_description(voice_description)

        segments: list[tuple[bytes, float]] = []
        for index, chunk in enumerate(chunks):
            seconds = min(audio_utils.estimate_seconds(chunk, speed), config.MAX_GEN_SECONDS)
            wav_bytes, _info = self._request(chunk, voice=voice, seconds=seconds,
                                             instruct=instruct, seed=(seed or 0) + index)
            if voice:
                wav_bytes, _info = self._guard_echo(
                    wav_bytes, _info, voice, chunk, seconds=seconds,
                    instruct=instruct, seed=(seed or 0) + index,
                )
            segments.append((wav_bytes, audio_utils.pause_after(chunk) * config.CHUNK_GAP_SCALE))

        wav_bytes, _rate = audio_utils.join_audio(segments)
        info = audio_utils.probe_wav_bytes(wav_bytes)
        notes.append(f"长文本已切分为 {len(chunks)} 段（每段 ≤ {limit:.0f} 秒）以保证语速自然")
        return SynthesisResult(
            wav_bytes=wav_bytes,
            sample_rate=int(info["sample_rate"]),
            duration=float(info["duration"]),
            engine=self.name,
            mode=("zero_shot_tts" if voice else "instruct_tts") + "+chunked",
            gen_seconds=round(float(info["duration"]), 3),
            notes=notes,
            warnings=warnings,
            elapsed=round(time.perf_counter() - started, 3),
        )

    # ------------------------------------------------------------------- editing
    def edit(self, instruction: str, source_audio: Path, *,
             duration_seconds: float | None = None, seed: int | None = None,
             variant: str | None = None) -> SynthesisResult:
        if self.cli_engine is None:
            raise EngineUnavailable("编辑功能需要 CLI 引擎，但当前未初始化")
        return self.cli_engine.edit(instruction, source_audio,
                                    duration_seconds=duration_seconds, seed=seed, variant=variant)


# ------------------------------------------------------------------------ factory
def clean_description(description: str, limit: int = MAX_VOICE_DESC_CHARS) -> tuple[str, bool]:
    """Make a voice description safe for the instruct template.

    Quotes break the ``描述: "…", 内容: "…"`` structure and are the main reason the
    model sometimes reads the description aloud, so they are stripped. Over-long
    descriptions are truncated because they invite the model to keep going.
    """
    if not description:
        return "", False
    cleaned = description.replace("\r", " ").replace("\n", " ")
    for ch in ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019", "\u300c", "\u300d"):
        cleaned = cleaned.replace(ch, "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    if len(cleaned) > limit:
        return cleaned[:limit].rstrip(), True
    return cleaned, False


def _torch_report() -> dict:
    if importlib.util.find_spec("torch") is None:
        return {"available": False, "reason": "torch not installed in this environment"}
    try:
        import torch

        report = {
            "available": True,
            "version": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["device_name"] = props.name
            report["vram_total_gib"] = round(props.total_memory / 1024 ** 3, 2)
        return report
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def preflight() -> dict:
    """Collect everything that decides whether real AuK inference can run."""
    assets = config.find_auk_assets()
    torch_info = _torch_report()
    missing: list[str] = []
    base_ready = bool(assets["base_ckpt"] and assets["base_config"])
    flash_ready = bool(assets["flash_ckpt"] and assets["flash_config"])
    if not (base_ready or flash_ready):
        missing.append(
            "AuK checkpoint + config (ckpts/AuK/auk_base.safetensors | ckpts/AuK-Flash/auk_flash.safetensors)"
        )
    if assets["qwen"] is None:
        missing.append("ckpts/Qwen2.5-Omni-3B (text + audio encoder)")
    if not torch_info.get("available"):
        missing.append("torch/torchaudio/transformers runtime")
    return {
        "ckpt_root": str(assets["ckpt_root"]),
        "assets": {k: (str(v) if v is not None else None) for k, v in assets.items()},
        "variants": {"base_ready": base_ready, "flash_ready": flash_ready},
        "torch": torch_info,
        "missing": missing,
        "ready": not missing,
    }


def soxr_chain_report() -> dict:
    """Actually load libsoxr.dll, so a broken dependency chain fails loud and early.

    libsoxr (MSYS2 UCRT build) -> libgomp-1.dll -> libwinpthread-1.dll. Missing any
    link makes the resident server answer ``requires Soxr HQ resampling`` at request
    time, which is indistinguishable from a transient glitch; probing here turns it
    into a startup-time, actionable error.
    """
    dll = config.AUDIO_CPP_DIR / config.SOXR_LIB_NAME
    deps = list(config.SOXR_DEPS)
    report = {
        "dll": str(dll),
        "present": dll.is_file(),
        "deps": {name: (config.AUDIO_CPP_DIR / name).is_file() for name in deps},
        "loadable": False,
        "error": None,
    }
    if not dll.is_file():
        report["error"] = "libsoxr.dll 缺失"
        return report
    missing = [name for name, ok in report["deps"].items() if not ok]
    if missing:
        report["error"] = "缺少依赖 " + ", ".join(missing)
        return report
    try:
        import ctypes
        import os as _os

        loader = ctypes.WinDLL if config.IS_WINDOWS else ctypes.CDLL
        if config.IS_WINDOWS:
            handle = _os.add_dll_directory(str(config.AUDIO_CPP_DIR))
            try:
                loader(str(dll))
                report["loadable"] = True
            finally:
                handle.close()
        else:
            loader(str(dll))
            report["loadable"] = True
    except Exception as exc:  # noqa: BLE001 - surfaced in /health
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def gguf_preflight(variant: str = "flash") -> dict:
    """Check the audio.cpp + GGUF backend."""
    cli = config.AUDIO_CPP_DIR / config.AUDIO_CPP_CLI_NAME
    model_dir = config.GGUF_MODEL_DIR
    files = config.GGUF_COMPONENTS.get(variant, {})
    required = {
        "cli": cli,
        "generator": model_dir / files.get("generator", "-"),
        "config": model_dir / files.get("config", "-"),
        "qwen": model_dir / config.GGUF_QWEN,
        "vae": model_dir / config.GGUF_VAE,
        "tokenizer": model_dir / "tokenizer" / "tokenizer.json",
        # not shipped in the release zip; a msvcrt-based build corrupts the heap at
        # teardown, so this must be the UCRT variant plus its OpenMP runtime
        "libsoxr": cli.parent / config.SOXR_LIB_NAME,
    }
    required.update({name: cli.parent / name for name in config.SOXR_DEPS})
    missing = [f"{name} ({path})" for name, path in required.items() if not path.exists()]
    soxr = soxr_chain_report()
    if not missing and not soxr["loadable"]:
        missing.append(f"libsoxr 无法加载：{soxr['error']}")
    cuda_bin = config.cuda_bin_dir()
    if config.GGUF_BACKEND == "cuda" and cuda_bin is None and "cli" not in [m.split(" ")[0] for m in missing]:
        missing.append("CUDA 12.x runtime dir (cufft64_11.dll / cudart64_12.dll not found)")
    return {
        "cli": str(cli),
        "model_dir": str(model_dir),
        "variant": variant,
        "backend": config.GGUF_BACKEND,
        "cuda_bin": str(cuda_bin) if cuda_bin else None,
        "required": {name: str(path) for name, path in required.items()},
        "soxr": soxr,
        "missing": missing,
        "ready": not missing,
    }


def build_engine(mode: str = "auto", *, device: str = "cuda", dtype: str = "bf16",
                 cpu_offload: bool = False, prefer: str = "flash", preload: bool = False,
                 gguf_variant: str | None = None, gguf_backend: str | None = None,
                 mem_saver: bool = False, tts_backend: str = "server",
                 server_url: str | None = None, server_autostart: bool | None = None):
    """Return ``(engine, requested_mode, fallback_reason)``.

    ``mode``:
      * ``auto`` — GGUF (audio.cpp) when ready, else the Python AuK stack, else mock
      * ``gguf`` — audio.cpp + GGUF only, fail fast
      * ``auk``  — Python torch stack only, fail fast
      * ``mock`` — placeholder audio, no model
    """
    mode = (mode or "auto").lower()
    if mode == "mock":
        return MockEngine("已按配置强制使用占位音引擎"), mode, None

    variant = gguf_variant or config.GGUF_VARIANT
    backend = gguf_backend or config.GGUF_BACKEND

    resolved_backend = (tts_backend or config.TTS_BACKEND).lower()

    def _gguf_engine() -> AukGgufEngine:
        return AukGgufEngine(
            config.AUDIO_CPP_DIR / config.AUDIO_CPP_CLI_NAME,
            config.GGUF_MODEL_DIR,
            variant=variant,
            backend=backend,
            mem_saver=mem_saver,
            variant_files=config.GGUF_COMPONENTS,
            edit_variant=config.GGUF_EDIT_VARIANT,
        )

    # --- resident server backend (fast path, used for live streaming) ----------
    if mode in ("auto", "server", "gguf") and resolved_backend == "server" and mode != "mock":
        try:
            proc, spawned = audioserver.ensure(server_url, autostart=server_autostart)
        except Exception as exc:
            reason = f"常驻服务不可用：{exc}"
            if mode == "server":
                raise EngineUnavailable(reason) from exc
            logger.warning("%s — 回退到 CLI 引擎", reason)
        else:
            cli_engine = None
            try:  # editing still runs through the CLI binary
                cli_engine = _gguf_engine()
            except Exception as exc:  # noqa: BLE001 - editing is optional
                logger.warning("CLI 引擎初始化失败，/edit 将不可用：%s", exc)

            engine = AukServerEngine(
                url=server_url or config.SERVER_URL,
                model_id=config.SERVER_MODEL_ID,
                voice_dir=config.VOICE_LIBRARY,
                cli_engine=cli_engine,
                server_process=proc,
                server_spawned=spawned,
            )
            if config.SERVER_WARMUP:
                # warm both routes: instruct (no reference) and voice (reference audio),
                # otherwise the first real request of each kind still pays a cold start
                try:
                    ok, took = audioserver.warmup(None, url=engine.url)
                    if ok:
                        logger.info("预热（Instruct 路径）完成，用时 %.2fs", took)
                    else:
                        logger.warning("预热（Instruct 路径）失败，用时 %.2fs", took)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("预热（Instruct）异常（不影响使用）：%s", exc)
                voices = sorted(config.VOICE_LIBRARY.glob("*.wav"))
                if voices:
                    name = voices[0].stem
                    for attempt in range(1, 3):  # a fresh process almost always loads soxr
                        try:
                            ok, took = audioserver.warmup(name, url=engine.url)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("预热（音色路径）异常：%s", exc)
                            ok, took = False, 0.0
                        if ok:
                            logger.info("预热（音色路径 %s）完成，用时 %.2fs", name, took)
                            break
                        logger.warning("预热（音色路径 %s）失败（第 %d/2 次），重启常驻服务后重试",
                                       name, attempt)
                        try:
                            audioserver.stop(proc)
                            proc, spawned = audioserver.ensure(engine.url)
                            engine.server_process, engine.server_spawned = proc, spawned
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("重启常驻服务失败：%s", exc)
                            break
            return engine, mode, None

    if mode in ("auto", "gguf"):
        gguf_report = gguf_preflight(variant)
        if gguf_report["ready"]:
            logger.info("engine: gguf (audiocpp_cli, variant=%s, backend=%s)", variant, backend)
            return _gguf_engine(), mode, None
        reason = "GGUF backend unavailable: missing " + "; ".join(gguf_report["missing"])
        if mode == "gguf":
            raise EngineUnavailable(reason)
        logger.warning("%s — falling back", reason)

    report = preflight()
    if report["missing"]:
        reason = "AuK inference unavailable: missing " + ", ".join(report["missing"])
        if mode == "auk":
            raise EngineUnavailable(reason)
        return MockEngine(reason), mode, reason

    assets = report["assets"]
    ckpt = assets["flash_ckpt"] if prefer == "flash" and assets["flash_ckpt"] else assets["base_ckpt"]
    cfg = assets["flash_config"] if ckpt == assets["flash_ckpt"] and assets["flash_config"] else assets["base_config"]
    if ckpt is None or cfg is None:
        reason = "AuK inference unavailable: no usable checkpoint/config pair"
        if mode == "auk":
            raise EngineUnavailable(reason)
        return MockEngine(reason), mode, reason

    engine = AukEngine(Path(ckpt), Path(cfg), assets["qwen"] and Path(assets["qwen"]),
                       device=device, dtype=dtype, cpu_offload=cpu_offload)
    if preload:
        try:
            engine.load()
        except EngineUnavailable as exc:
            if mode == "auk":
                raise
            return MockEngine(str(exc)), mode, str(exc)
    return engine, mode, None
