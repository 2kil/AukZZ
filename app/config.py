"""Path and runtime configuration for the AuK speech service.

All paths are resolved inside the project root (D:/GitSum/AukZZ); nothing is
written outside of it. Environment variables can override any location.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = os.name == "nt"

# --- data locations ---------------------------------------------------------
ROLE_LIBRARY_DIR = Path(os.getenv("AUK_ROLE_DIR", ROOT / "role")).resolve()
DATA_DIR = Path(os.getenv("AUK_DATA_DIR", ROOT / "data")).resolve()
ROLE_STORE_DIR = DATA_DIR / "roles"
ROLE_STORE_AUDIO = ROLE_STORE_DIR / "audio"
ROLE_STORE_INDEX = ROLE_STORE_DIR / "index.json"
OUTPUT_DIR = Path(os.getenv("AUK_OUTPUT_DIR", DATA_DIR / "outputs")).resolve()
TMP_DIR = OUTPUT_DIR / "_tmp"

# --- AuK reference implementation (read-only copy inside the workspace) -----
AUK_SRC = Path(os.getenv("AUK_SRC", ROOT / "docs" / "AuK" / "src")).resolve()

_CKPT_CANDIDATES = [
    Path(os.getenv("AUK_CKPT_DIR")) if os.getenv("AUK_CKPT_DIR") else None,
    ROOT / "docs" / "AuK" / "ckpts",
    ROOT / "ckpts",
]


def ckpt_root() -> Path:
    """First existing checkpoint root, else the documented default."""
    for candidate in _CKPT_CANDIDATES:
        if candidate and Path(candidate).is_dir():
            return Path(candidate).resolve()
    return (ROOT / "docs" / "AuK" / "ckpts").resolve()


def _existing(*relative: str) -> Path | None:
    base = ckpt_root()
    for rel in relative:
        path = base / rel
        if path.exists():
            return path
    return None


def find_auk_assets() -> dict:
    """Locate the checkpoint / config / text-encoder paths required by AukInfer."""
    return {
        "ckpt_root": ckpt_root(),
        "flash_ckpt": _existing("AuK-Flash/auk_flash.safetensors", "auk_flash.safetensors"),
        "base_ckpt": _existing("AuK/auk_base.safetensors", "auk_base.safetensors"),
        "flash_config": _existing("AuK-Flash/config.yaml"),
        "base_config": _existing("AuK/config.yaml", "config.yaml"),
        "qwen": _existing("Qwen2.5-Omni-3B"),
    }


# --- generation defaults ----------------------------------------------------
TARGET_SAMPLE_RATE = int(os.getenv("AUK_SAMPLE_RATE", "24000"))
REF_TRIM_SECONDS = float(os.getenv("AUK_REF_TRIM_SECONDS", "10"))
# AuK's source/reference + target sequence budget is ~30 s; keep headroom.
MAX_GEN_SECONDS = float(os.getenv("AUK_MAX_GEN_SECONDS", "20"))
# long scripts are split into chunks that each fit this budget (keeps speech rate natural)
MAX_CHUNK_SECONDS = float(os.getenv("AUK_MAX_CHUNK_SECONDS", "15"))
CHUNK_GAP_SCALE = float(os.getenv("AUK_CHUNK_GAP_SCALE", "1.0"))

DEFAULT_ENGINE = os.getenv("AUK_ENGINE", "auto")
DEFAULT_HOST = os.getenv("AUK_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("AUK_PORT", "8080"))

# --- audio.cpp / GGUF backend ----------------------------------------------
AUDIO_CPP_DIR = Path(os.getenv("AUDIO_CPP_DIR", ROOT / "vendor" / "audio.cpp-cu124")).resolve()
AUDIO_CPP_CLI_NAME = os.getenv(
    "AUDIO_CPP_CLI", "audiocpp_cli.exe" if IS_WINDOWS else "audiocpp_cli"
)
AUDIO_CPP_SERVER_NAME = os.getenv(
    "AUDIO_CPP_SERVER", "audiocpp_server.exe" if IS_WINDOWS else "audiocpp_server"
)
GGUF_MODEL_DIR = Path(os.getenv("AUK_GGUF_DIR", ROOT / "models" / "AuK-GGUF")).resolve()
GGUF_VARIANT = os.getenv("AUK_GGUF_VARIANT", "flash")  # flash (4-step) | base
GGUF_EDIT_VARIANT = os.getenv("AUK_GGUF_EDIT_VARIANT", "base")  # base keeps editing faithful
GGUF_BACKEND = os.getenv("AUK_GGUF_BACKEND", "cuda")  # cuda | vulkan | cpu
GGUF_TIMEOUT = float(os.getenv("AUK_GGUF_TIMEOUT", "900"))

# --- resident audio.cpp server (fast path for live use) --------------------
TTS_BACKEND = os.getenv("AUK_TTS_BACKEND", "server")        # server | cli
SERVER_URL = os.getenv("AUK_SERVER_URL", "http://127.0.0.1:8123")
SERVER_MODEL_ID = os.getenv("AUK_SERVER_MODEL", "auk-flash")
SERVER_CONFIG = Path(os.getenv("AUK_SERVER_CONFIG", ROOT / "vendor" / "server.json")).resolve()
VOICE_LIBRARY = Path(os.getenv("AUK_VOICE_DIR", ROOT / "models" / "voice-library")).resolve()
SERVER_AUTOSTART = os.getenv("AUK_SERVER_AUTOSTART", "1") not in ("0", "false", "False")
SERVER_READY_TIMEOUT = float(os.getenv("AUK_SERVER_READY_TIMEOUT", "120"))
SERVER_WARMUP = os.getenv("AUK_SERVER_WARMUP", "1") not in ("0", "false", "False")

GGUF_COMPONENTS = {
    "flash": {
        "generator": "auk-flash-q8_0.gguf",
        "config": "config/auk-flash.yaml",
    },
    "base": {
        "generator": "auk-base-q8_0.gguf",
        "config": "config/auk-base.yaml",
    },
}
# libsoxr is loaded at runtime by audio.cpp (AuK needs HQ resampling for the
# reference audio). Windows build: libsoxr.dll + libgomp-1.dll + libwinpthread-1.dll.
# Linux build: libsoxr.so.0 from the distro ("apt install libsoxr0"), one file only.
SOXR_LIB_NAME = "libsoxr.dll" if IS_WINDOWS else "libsoxr.so.0"
SOXR_DEPS = ("libgomp-1.dll", "libwinpthread-1.dll") if IS_WINDOWS else ()

GGUF_QWEN = "qwen2.5-omni-3b-q8_0.gguf"
GGUF_VAE = "auk-vae-f32.gguf"


def cuda_bin_dir() -> Path | None:
    """Locate the CUDA runtime dir the bundled audio.cpp build expects.

    The Windows cu124 build needs ``cufft64_11.dll`` / ``cudart64_12.dll`` on PATH;
    the Ubuntu build (cuda12.8) needs ``libcufft.so.11`` on LD_LIBRARY_PATH, either
    from the CUDA toolkit or from the extracted cudart archive.
    """
    candidates: list[Path] = []
    probes: tuple[str, ...]
    if IS_WINDOWS:
        probes = ("cufft64_11.dll",)
        for env_name in ("CUDA_HOME", "CUDA_PATH"):
            value = os.getenv(env_name)
            if value:
                candidates.append(Path(value) / "bin")
        toolkit_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        if toolkit_root.is_dir():
            candidates.extend(sorted(toolkit_root.glob("v12*/bin"), reverse=True))
    else:
        probes = ("libcufft.so.11", "libcufft.so")
        for env_name in ("CUDA_HOME", "CUDA_PATH"):
            value = os.getenv(env_name)
            if value:
                candidates.extend([Path(value) / "lib64", Path(value) / "lib"])
        candidates.append(AUDIO_CPP_DIR)  # cudart archive often extracted here

    for candidate in candidates:
        if any((candidate / probe).exists() for probe in probes):
            return candidate
    return None


def ensure_dirs() -> None:
    for path in (ROLE_STORE_AUDIO, OUTPUT_DIR, TMP_DIR):
        path.mkdir(parents=True, exist_ok=True)
