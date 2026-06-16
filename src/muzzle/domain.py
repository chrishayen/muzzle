from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TTSQuality = Literal["fast", "balanced", "high", "cpu-smooth"]

TTS_QUALITY_PROFILES = {
    "fast": {
        "chunk_tokens": 24,
        "crossfade_ms": 12.0,
        "temperature": 0.8,
        "top_p": 0.95,
    },
    "balanced": {
        "chunk_tokens": 48,
        "crossfade_ms": 24.0,
        "temperature": 0.72,
        "top_p": 0.92,
    },
    "high": {
        "chunk_tokens": 64,
        "crossfade_ms": 36.0,
        "temperature": 0.65,
        "top_p": 0.9,
    },
    "cpu-smooth": {
        "chunk_tokens": 64,
        "crossfade_ms": 40.0,
        "temperature": 0.8,
        "top_p": 0.95,
    },
}


@dataclass(frozen=True)
class VoiceInfo:
    voice_id: str
    name: str
    source: Literal["builtin", "managed"]
    created_at: str | None = None
    sample_rate: int | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True)
class VoiceRecord:
    voice_id: str
    name: str
    source_filename: str
    reference_audio_path: Path
    directory: Path
    created_at: str
    sample_rate: int | None = None
    duration_seconds: float | None = None

    def to_info(self) -> VoiceInfo:
        return VoiceInfo(
            voice_id=self.voice_id,
            name=self.name,
            source="managed",
            created_at=self.created_at,
            sample_rate=self.sample_rate,
            duration_seconds=self.duration_seconds,
        )


@dataclass(frozen=True)
class TTSOptions:
    request_id: str
    text: str
    voice_id: str
    quality: TTSQuality = "balanced"
    chunk_tokens: int = 48
    crossfade_ms: float = 24.0
    temperature: float = 0.72
    top_p: float = 0.92
    top_k: int = 1000
    repetition_penalty: float = 1.2
    max_gen_len: int = 1000


@dataclass(frozen=True)
class TTSChunk:
    request_id: str
    audio: bytes
    sample_rate: int
    index: int
    is_final: bool
    start_sample: int
    end_sample: int
    generated_tokens: int
    watermarked: bool = False


@dataclass(frozen=True)
class TranscriptEvent:
    kind: Literal["speech_started", "speech_stopped", "partial", "final"]
    text: str = ""
    start_ms: int | None = None
    end_ms: int | None = None
    language: str | None = None
