"""Lifecycle helper for the resident audio.cpp server (``audiocpp_server.exe``).

The speech service keeps its HTTP contract (``/speech``) but forwards synthesis to
this long-lived process, which holds the AuK weights in VRAM. That removes the
~6-7 s per-request model load of the CLI path: steady state becomes ~0.3 s.

The server is started on demand (if reachable already, it is reused) and stopped
again when the speech service shuts down, unless it was already running.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config, audio as audio_utils

logger = logging.getLogger("auk.service.server")


def _base(url: str | None = None) -> str:
    return (url or config.SERVER_URL).rstrip("/")


def request_json(url: str, payload: dict, *, timeout: float = 300.0) -> tuple[int, dict | bytes]:
    """POST JSON and return (status, parsed-json-or-bytes)."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:  # 4xx/5xx still carry a JSON body
        raw = exc.read()
        status = exc.code
    except urllib.error.URLError as exc:
        raise ConnectionError(f"无法连接常驻服务 {url}：{exc.reason}") from exc
    if raw[:4] == b"RIFF":
        return status, raw
    try:
        return status, json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return status, raw


def models(url: str | None = None, *, timeout: float = 5.0) -> list[dict]:
    try:
        with urllib.request.urlopen(f"{_base(url)}/v1/models", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("data", [])
    except Exception:
        return []


def is_ready(url: str | None = None, model_id: str | None = None) -> bool:
    want = model_id or config.SERVER_MODEL_ID
    for item in models(url):
        if item.get("id") == want:
            return bool(item.get("loaded"))
    return False


def _port_open(url: str) -> bool:
    try:
        host = url.split("//", 1)[-1].split("/", 1)[0]
        hostname, _, port = host.partition(":")
        with socket.create_connection((hostname, int(port or 80)), timeout=1.0):
            return True
    except Exception:
        return False


def _env() -> dict:
    env = os.environ.copy()
    cuda_bin = config.cuda_bin_dir()
    if cuda_bin is not None:
        # Windows resolves DLLs from PATH, Linux from LD_LIBRARY_PATH
        key = "PATH" if config.IS_WINDOWS else "LD_LIBRARY_PATH"
        env[key] = f"{cuda_bin}{os.pathsep}{env.get(key, '')}"
    return env


def spawn(config_path: Path | None = None) -> subprocess.Popen:
    cfg = Path(config_path or config.SERVER_CONFIG)
    cli_dir = config.AUDIO_CPP_DIR
    exe = cli_dir / config.AUDIO_CPP_SERVER_NAME
    if not exe.is_file():
        raise FileNotFoundError(f"{config.AUDIO_CPP_SERVER_NAME} 不存在：{exe}")
    if not cfg.is_file():
        raise FileNotFoundError(f"常驻服务配置不存在：{cfg}")
    log_path = config.TMP_DIR / "audiocpp_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "ab")  # noqa: SIM115 - kept open for the child's lifetime
    logger.info("启动常驻服务：%s --config %s（日志 %s）", exe.name, cfg, log_path)
    proc = subprocess.Popen(
        [str(exe), "--config", str(cfg)],
        cwd=str(cli_dir),
        env=_env(),
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    proc._auk_log = handle  # type: ignore[attr-defined]
    return proc


def wait_ready(url: str | None = None, *, timeout: float | None = None) -> bool:
    deadline = time.time() + (timeout if timeout is not None else config.SERVER_READY_TIMEOUT)
    while time.time() < deadline:
        if is_ready(url):
            return True
        time.sleep(1.0)
    return False


def ensure(url: str | None = None, *, autostart: bool | None = None,
           model_id: str | None = None) -> tuple[subprocess.Popen | None, bool]:
    """Make sure a loaded model is available.

    Returns ``(process, spawned)``; ``process`` is only set when this call started it.
    """
    target = _base(url)
    want = model_id or config.SERVER_MODEL_ID
    if is_ready(target, want):
        logger.info("复用已运行的常驻服务 %s（model=%s）", target, want)
        return None, False

    auto = config.SERVER_AUTOSTART if autostart is None else autostart
    if not auto:
        raise ConnectionError(f"常驻服务未就绪且未开启自动启动：{target}")

    if _port_open(target):
        # something is listening but the model is not loaded yet (still starting)
        if wait_ready(target):
            return None, False
        raise ConnectionError(f"常驻服务 {target} 有进程监听但模型未就绪")

    proc = spawn()
    if not wait_ready(target):
        stop(proc)
        raise ConnectionError(
            f"常驻服务启动超时（{config.SERVER_READY_TIMEOUT:.0f}s），请查看 {config.TMP_DIR / 'audiocpp_server.log'}"
        )
    return proc, True


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    logger.info("停止常驻服务 pid=%s", proc.pid)
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    handle = getattr(proc, "_auk_log", None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


def warmup(voice: str | None = None, *, url: str | None = None,
           model_id: str | None = None) -> tuple[bool, float]:
    """One throwaway generation so the first real request is not cold (~5 s)."""
    started = time.perf_counter()
    options: dict = {"duration_sec": 1.0}
    payload: dict = {
        "model": model_id or config.SERVER_MODEL_ID,
        "input": audio_utils.normalize_text("预热"),
        "options": options,
    }
    if voice:
        payload["voice"] = voice
    else:  # instruct route keeps the warmup independent of any role
        options["instruct"] = "一位成年女性，语气平稳，吐字清晰"
    status, body, ok = 0, None, False
    for attempt in range(1, 4):
        status, body = request_json(f"{_base(url)}/v1/audio/speech", payload, timeout=180.0)
        if status == 200:
            ok = True
            break
        detail = str(body)[:200]
        logger.warning("预热失败（第 %d/3 次）status=%s body=%s", attempt, status, detail)
        if "soxr" not in detail.lower() and "resampling" not in detail.lower():
            break
        time.sleep(0.8)
    return ok, round(time.perf_counter() - started, 2)
