from __future__ import annotations

from typing import AsyncIterator, Protocol

from ..domain import TTSChunk, TTSOptions, TranscriptEvent, VoiceInfo, VoiceRecord


class TTSAdapter(Protocol):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    def default_voice(self) -> VoiceInfo | None:
        ...

    async def prepare_voice(self, record: VoiceRecord) -> None:
        ...

    def synthesize_stream(
        self,
        options: TTSOptions,
        voice: VoiceRecord | None,
    ) -> AsyncIterator[TTSChunk]:
        ...


class STTStream(Protocol):
    async def receive_audio(self, frame: bytes, *, sample_rate: int) -> list[TranscriptEvent]:
        ...

    async def commit(self) -> list[TranscriptEvent]:
        ...


class STTAdapter(Protocol):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    def create_stream(self, *, language: str | None) -> STTStream:
        ...

