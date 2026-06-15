from __future__ import annotations

import asyncio

from ..audio import INPUT_SAMPLE_RATE, pcm16_duration_ms, pcm16_has_energy, silence_pcm16
from ..config import Settings
from ..domain import TTSChunk, TTSOptions, TranscriptEvent, VoiceInfo, VoiceRecord


class FakeTTSAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def default_voice(self) -> VoiceInfo:
        return VoiceInfo(voice_id="default", name="Fake default", source="builtin")

    async def prepare_voice(self, record: VoiceRecord) -> None:
        if record.duration_seconds is not None and record.duration_seconds < 5.0:
            raise ValueError("reference audio must be at least 5 seconds long")
        (record.directory / "fake_conds.json").write_text('{"prepared": true}\n')

    async def synthesize_stream(self, options: TTSOptions, voice: VoiceRecord | None):
        sample_rate = 24_000
        chunk_ms = 120
        start_sample = 0
        for index in range(2):
            await asyncio.sleep(0)
            audio = silence_pcm16(sample_rate, chunk_ms)
            end_sample = start_sample + len(audio) // 2
            yield TTSChunk(
                request_id=options.request_id,
                audio=audio,
                sample_rate=sample_rate,
                index=index,
                is_final=index == 1,
                start_sample=start_sample,
                end_sample=end_sample,
                generated_tokens=(index + 1) * options.chunk_tokens,
                watermarked=False,
            )
            start_sample = end_sample


class FakeSTTAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def create_stream(self, *, language: str | None):
        return FakeSTTStream(
            language=language,
            partial_interval_ms=self.settings.stt_partial_interval_ms,
            silence_finalize_ms=self.settings.stt_silence_finalize_ms,
        )


class FakeSTTStream:
    def __init__(self, *, language: str | None, partial_interval_ms: int, silence_finalize_ms: int):
        self.language = language
        self.partial_interval_ms = partial_interval_ms
        self.silence_finalize_ms = silence_finalize_ms
        self.active = False
        self.buffer = bytearray()
        self.elapsed_ms = 0.0
        self.silence_ms = 0.0
        self.next_partial_ms = float(partial_interval_ms)
        self.utterance_index = 0

    async def receive_audio(self, frame: bytes, *, sample_rate: int) -> list[TranscriptEvent]:
        if sample_rate != INPUT_SAMPLE_RATE:
            raise ValueError("fake STT only accepts 16 kHz audio")

        events: list[TranscriptEvent] = []
        duration_ms = pcm16_duration_ms(frame, sample_rate=sample_rate)
        has_speech = pcm16_has_energy(frame)
        self.elapsed_ms += duration_ms

        if has_speech:
            if not self.active:
                self.active = True
                self.silence_ms = 0.0
                events.append(TranscriptEvent(kind="speech_started", start_ms=int(self.elapsed_ms - duration_ms)))
            self.buffer.extend(frame)
            self.silence_ms = 0.0
            if len(self.buffer) // 2 / sample_rate * 1000.0 >= self.next_partial_ms:
                events.append(
                    TranscriptEvent(
                        kind="partial",
                        text=f"partial transcript {self.utterance_index + 1}",
                        start_ms=0,
                        end_ms=int(self.elapsed_ms),
                        language=self.language,
                    )
                )
                self.next_partial_ms += self.partial_interval_ms
            return events

        if self.active:
            self.silence_ms += duration_ms
            self.buffer.extend(frame)
            if self.silence_ms >= self.silence_finalize_ms:
                events.extend(await self._finalize(end_ms=int(self.elapsed_ms)))
        return events

    async def commit(self) -> list[TranscriptEvent]:
        if not self.buffer and not self.active:
            return []
        return await self._finalize(end_ms=int(self.elapsed_ms))

    async def _finalize(self, *, end_ms: int) -> list[TranscriptEvent]:
        self.utterance_index += 1
        text = f"final transcript {self.utterance_index}"
        self.active = False
        self.buffer.clear()
        self.silence_ms = 0.0
        self.next_partial_ms = float(self.partial_interval_ms)
        return [
            TranscriptEvent(kind="speech_stopped", end_ms=end_ms),
            TranscriptEvent(kind="final", text=text, start_ms=0, end_ms=end_ms, language=self.language),
        ]
