"""Offline checks for the resident-server request payload.

The resident backend forwards ``input`` to audio.cpp verbatim, so the instruction
wrapping done by the CLI path must be reproduced here. Missing it made AuK treat the
target sentence as an instruction and returned audio unrelated to the text
(measured correlation 0.09 against the CLI baseline, vs 1.000 when wrapped).

Run: .venv/Scripts/python.exe tests/test_payload.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import audio as audio_utils  # noqa: E402
from app import audioserver  # noqa: E402
from app.engine import AukServerEngine  # noqa: E402

REF = ROOT / "role" / "安若柔.wav"
TEXT = "今天天气非常好，我们出去走走吧。"

captured: list[dict] = []


def fake_request_json(url, payload, *, timeout=300.0):  # noqa: ANN001, ARG001
    captured.append(payload)
    wav = audio_utils.tone_wav_bytes(1.0, 24000)
    return 200, wav


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    audioserver.request_json = fake_request_json  # type: ignore[assignment]
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp:
        engine = AukServerEngine(url="http://127.0.0.1:9", model_id="auk-flash",
                                 voice_dir=Path(tmp))

        captured.clear()
        engine.synthesize(TEXT, ref_audio=REF, voice_description=None, gen_seconds=3.0,
                          speed=1.0, seed=42, nfe=32, cfg_strength=2.0,
                          role_name="安若柔")
        payload = captured[-1]
        expected = f'Say the following with the same voice: "{TEXT}"'
        results.append(check(
            "zero-shot 请求带上完整指令模板",
            payload.get("input") == expected,
            repr(payload.get("input"))[:70],
        ))
        results.append(check("zero-shot 请求带 voice 字段", "voice" in payload,
                             str(payload.get("voice"))))
        results.append(check("zero-shot 请求不额外带 instruct", "instruct" not in payload["options"]))

        captured.clear()
        engine.synthesize(TEXT, ref_audio=None, voice_description="一位温柔的女声",
                          gen_seconds=3.0, speed=1.0, seed=42, nfe=32, cfg_strength=2.0,
                          role_name=None)
        payload = captured[-1]
        results.append(check("instruct 请求保持裸文本", payload.get("input") == TEXT,
                             repr(payload.get("input"))[:70]))
        results.append(check("instruct 请求把描述放进 options.instruct",
                             payload["options"].get("instruct") == "一位温柔的女声"))
        results.append(check("instruct 请求不带 voice", "voice" not in payload))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
