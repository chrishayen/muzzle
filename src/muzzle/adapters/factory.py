from __future__ import annotations

from ..config import Settings
from .base import STTAdapter, TTSAdapter
from .fake import FakeSTTAdapter, FakeTTSAdapter


def build_tts_adapter(settings: Settings) -> TTSAdapter:
    if settings.model_backend == "fake":
        return FakeTTSAdapter(settings)
    if settings.model_backend != "real":
        raise ValueError(f"unsupported MUZZLE_MODEL_BACKEND={settings.model_backend!r}")
    from .chatterbox import ChatterboxTTSAdapter

    return ChatterboxTTSAdapter(settings)


def build_stt_adapter(settings: Settings) -> STTAdapter:
    if settings.model_backend == "fake":
        return FakeSTTAdapter(settings)
    if settings.model_backend != "real":
        raise ValueError(f"unsupported MUZZLE_MODEL_BACKEND={settings.model_backend!r}")
    from .whisper import WhisperSTTAdapter

    return WhisperSTTAdapter(settings)

