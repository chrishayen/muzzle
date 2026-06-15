from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    model_backend: str = "real"
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    auth_token: str | None = None
    max_sessions: int = 4
    max_upload_mb: int = 100
    tts_device: str = "auto"
    whisper_model: str = "distil-large-v3"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    stt_partial_interval_ms: int = 500
    stt_silence_finalize_ms: int = 800
    stt_max_segment_seconds: float = 30.0
    stt_language: str | None = "en"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("MUZZLE_DATA_DIR", "data")),
            model_backend=os.getenv("MUZZLE_MODEL_BACKEND", "real"),
            host=os.getenv("MUZZLE_HOST", "127.0.0.1"),
            port=int(os.getenv("MUZZLE_PORT", "8000")),
            reload=os.getenv("MUZZLE_RELOAD", "false").lower() in {"1", "true", "yes"},
            auth_token=_optional_env("MUZZLE_AUTH_TOKEN"),
            max_sessions=int(os.getenv("MUZZLE_MAX_SESSIONS", "4")),
            max_upload_mb=int(os.getenv("MUZZLE_MAX_UPLOAD_MB", "100")),
            tts_device=os.getenv("MUZZLE_TTS_DEVICE", "auto"),
            whisper_model=os.getenv("MUZZLE_WHISPER_MODEL", "distil-large-v3"),
            whisper_device=os.getenv("MUZZLE_WHISPER_DEVICE", "auto"),
            whisper_compute_type=os.getenv("MUZZLE_WHISPER_COMPUTE_TYPE", "auto"),
            stt_partial_interval_ms=int(os.getenv("MUZZLE_STT_PARTIAL_INTERVAL_MS", "500")),
            stt_silence_finalize_ms=int(os.getenv("MUZZLE_STT_SILENCE_FINALIZE_MS", "800")),
            stt_max_segment_seconds=float(os.getenv("MUZZLE_STT_MAX_SEGMENT_SECONDS", "30")),
            stt_language=_optional_env("MUZZLE_STT_LANGUAGE") or "en",
        )

