"""Role (voice) library: built-in voices from ./role plus user-created roles."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from . import audio as audio_utils
from . import config


@dataclass
class Role:
    id: str
    name: str
    source: str  # "builtin" | "custom"
    audio_path: str
    duration: float
    sample_rate: int
    channels: int
    description: str = ""
    gender: str = ""
    language: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_public(self) -> dict:
        data = asdict(self)
        data["audio_url"] = f"/roles/{self.id}/audio"
        return data


def _stable_id(prefix: str, name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


class RoleStore:
    """Thread-safe role registry with JSON persistence for custom roles."""

    def __init__(self, library_dir: Path | None = None) -> None:
        self.library_dir = Path(library_dir or config.ROLE_LIBRARY_DIR)
        self.store_dir = config.ROLE_STORE_DIR
        self.audio_dir = config.ROLE_STORE_AUDIO
        self.index_path = config.ROLE_STORE_INDEX
        self._lock = threading.RLock()
        self._custom: dict[str, Role] = {}
        self._builtin: dict[str, Role] = {}
        self._builtin_warnings: list[str] = []
        config.ensure_dirs()
        self._load_custom()
        self.refresh_builtin()

    # ------------------------------------------------------------- persistence
    def _load_custom(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as exc:  # corrupted index must not block startup
            self._builtin_warnings.append(f"custom role index unreadable: {exc}")
            return
        for item in raw.get("roles", []):
            path = Path(item.get("audio_path", ""))
            if not path.is_file():
                continue
            self._custom[item["id"]] = Role(
                id=item["id"],
                name=item["name"],
                source="custom",
                audio_path=str(path),
                duration=float(item.get("duration", 0)),
                sample_rate=int(item.get("sample_rate", config.TARGET_SAMPLE_RATE)),
                channels=int(item.get("channels", 1)),
                description=item.get("description", ""),
                gender=item.get("gender", ""),
                language=item.get("language", ""),
                tags=list(item.get("tags", [])),
                created_at=item.get("created_at", ""),
            )

    def _save_custom(self) -> None:
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "roles": [asdict(role) for role in self._custom.values()],
        }
        self.store_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)

    # ---------------------------------------------------------------- built-in
    def refresh_builtin(self) -> None:
        """(Re)scan ./role for bundled reference audio."""
        with self._lock:
            self._builtin.clear()
            self._builtin_warnings = []
            if not self.library_dir.is_dir():
                self._builtin_warnings.append(f"role directory not found: {self.library_dir}")
                return
            for path in sorted(self.library_dir.glob("*.wav")):
                name = path.stem
                try:
                    info = audio_utils.probe_wav(path)
                except Exception as exc:
                    self._builtin_warnings.append(f"skipped {path.name}: {exc}")
                    continue
                role_id = _stable_id("builtin", name)
                self._builtin[role_id] = Role(
                    id=role_id,
                    name=name,
                    source="builtin",
                    audio_path=str(path.resolve()),
                    duration=info["duration"],
                    sample_rate=info["sample_rate"],
                    channels=info["channels"],
                    description=f"内置角色音色（来自 role/{path.name}）",
                    language="zh",
                    tags=["builtin"],
                )

    # ------------------------------------------------------------------ queries
    def list(self) -> list[Role]:
        with self._lock:
            return list(self._builtin.values()) + list(self._custom.values())

    def stats(self) -> dict:
        with self._lock:
            return {
                "builtin": len(self._builtin),
                "custom": len(self._custom),
                "library_dir": str(self.library_dir),
                "warnings": list(self._builtin_warnings),
            }

    def get(self, identifier: str | None) -> Role | None:
        if not identifier:
            return None
        with self._lock:
            if identifier in self._builtin:
                return self._builtin[identifier]
            if identifier in self._custom:
                return self._custom[identifier]
            for role in self.list():
                if role.name == identifier:
                    return role
            lowered = identifier.lower()
            for role in self.list():
                if role.name.lower() == lowered:
                    return role
            return None

    # ----------------------------------------------------------------- mutation
    def create(
        self,
        *,
        name: str,
        upload_path: Path,
        description: str = "",
        gender: str = "",
        language: str = "",
        tags: Iterable[str] = (),
        trim_seconds: float | None = None,
    ) -> Role:
        """Register a new role from a validated WAV file already on disk."""
        with self._lock:
            if self.get(name):
                raise ValueError(f"role name already exists: {name}")
            role_id = _stable_id("role", name)
            target = self.audio_dir / f"{role_id}.wav"
            limit = config.REF_TRIM_SECONDS if trim_seconds is None else trim_seconds
            # always normalise to mono PCM16 (handles float / multichannel uploads)
            info = audio_utils.trim_wav(upload_path, target, limit if limit and limit > 0 else 0)
            role = Role(
                id=role_id,
                name=name,
                source="custom",
                audio_path=str(target.resolve()),
                duration=info["duration"],
                sample_rate=info["sample_rate"],
                channels=info["channels"],
                description=description,
                gender=gender,
                language=language,
                tags=[t for t in tags if t],
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            self._custom[role.id] = role
            self._save_custom()
            return role

    def delete(self, role_id: str) -> bool:
        with self._lock:
            role = self._custom.pop(role_id, None)
            if role is None:
                return False
            audio_file = Path(role.audio_path)
            if self.audio_dir in audio_file.parents and audio_file.is_file():
                audio_file.unlink(missing_ok=True)
            self._save_custom()
            return True
