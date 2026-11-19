"""Endpoint smoke test (stdlib only, no extra dependencies).

Usage:
    .venv\\Scripts\\python.exe tests\\smoke_test.py --base http://127.0.0.1:8080

Exercises: /health, GET /roles, POST /roles (multipart upload), POST /speech
(zero-shot with a built-in role), /roles/{id}/audio, DELETE /roles/{id}.
Exits non-zero on the first failure.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_WAV = PROJECT_ROOT / "role" / "安若柔.wav"


def _request(base: str, path: str, *, method: str = "GET", body: bytes | None = None,
             headers: dict | None = None):
    req = urllib.request.Request(f"{base}{path}", data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()


def _multipart(fields: dict[str, str], file_field: str, filename: str, content_type: str,
               payload: bytes) -> tuple[bytes, str]:
    boundary = f"----auk{uuid.uuid4().hex}"
    buffer = io.BytesIO()
    for key, value in fields.items():
        buffer.write(f"--{boundary}\r\n".encode())
        buffer.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        buffer.write(f"{value}\r\n".encode())
    buffer.write(f"--{boundary}\r\n".encode())
    buffer.write(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode()
    )
    buffer.write(f"Content-Type: {content_type}\r\n\r\n".encode())
    buffer.write(payload)
    buffer.write(f"\r\n--{boundary}--\r\n".encode())
    return buffer.getvalue(), f"multipart/form-data; boundary={boundary}"


def _check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    status, _, raw = _request(base, "/health")
    health = json.loads(raw)
    _check("GET /health", status == 200 and health["status"] == "ok",
           f"engine={health['engine']['active']} roles={health['roles']}")
    if health["engine"]["active"] == "mock":
        print(f"       degraded: {health['engine']['reason']}")

    status, _, raw = _request(base, "/roles")
    roles = json.loads(raw)
    _check("GET /roles", status == 200 and roles["total"] >= 1, f"total={roles['total']}")

    builtin = next((r for r in roles["roles"] if r["source"] == "builtin"), None)
    _check("built-in role from ./role", builtin is not None,
           builtin["name"] if builtin else "none found")
    if builtin is None:
        sys.exit(1)

    payload = SAMPLE_WAV.read_bytes() if SAMPLE_WAV.is_file() else b""
    name = f"smoke-{uuid.uuid4().hex[:6]}"
    body, content_type = _multipart(
        {"name": name, "description": "smoke test", "language": "zh"},
        "ref_audio", f"{name}.wav", "audio/wav", payload,
    )
    status, _, raw = _request(base, "/roles", method="POST", body=body,
                              headers={"Content-Type": content_type})
    created = json.loads(raw)
    _check("POST /roles", status == 201 and created["name"] == name,
           f"id={created.get('id')} duration={created.get('duration')}s")

    payload_json = json.dumps({"text": "你好，这是一次接口冒烟测试。", "role_id": created["id"]}).encode()
    status, headers, raw = _request(base, "/speech", method="POST", body=payload_json,
                                    headers={"Content-Type": "application/json"})
    with wave.open(io.BytesIO(raw), "rb") as handle:
        duration = handle.getnframes() / handle.getframerate()
        params = f"{handle.getframerate()}Hz/{handle.getnchannels()}ch"
    _check("POST /speech", status == 200 and len(raw) > 1000,
           f"{len(raw)} bytes {params} {duration:.2f}s engine={headers.get('x-engine')}")

    status, headers, raw = _request(base, f"/roles/{created['id']}/audio")
    _check("GET /roles/{id}/audio", status == 200 and len(raw) > 1000, f"{len(raw)} bytes")

    edit_body, edit_ct = _multipart(
        {"instruction": "Raise the pitch by 2 semitones.", "seed": "42"},
        "audio", "src.wav", "audio/wav", payload,
    )
    status, headers, raw = _request(base, "/edit", method="POST", body=edit_body,
                                    headers={"Content-Type": edit_ct})
    with wave.open(io.BytesIO(raw), "rb") as handle:
        edit_dur = handle.getnframes() / handle.getframerate()
    _check("POST /edit", status == 200 and len(raw) > 1000,
           f"{len(raw)} bytes {edit_dur:.2f}s mode={headers.get('x-mode')}")

    status, _, raw = _request(base, f"/roles/{created['id']}", method="DELETE")
    _check("DELETE /roles/{id}", status == 200, json.loads(raw).get("deleted", ""))

    print("\nall smoke checks passed")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}")
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"cannot reach service: {exc.reason}")
        sys.exit(1)
