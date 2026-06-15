from __future__ import annotations

import anyio

from ..audio import INPUT_SAMPLE_RATE, pcm16_duration_ms, pcm16_to_float32
from ..config import Settings
from ..domain import TranscriptEvent


class WhisperSTTAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None

    async def start(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            if exc.name == "faster_whisper":
                raise RuntimeError("faster-whisper is not installed. Run `make sync-real` before `make dev`.") from exc
            raise

        device = _resolve_device(self.settings.whisper_device)
        compute_type = _resolve_compute_type(self.settings.whisper_compute_type, device)
        self.model = await anyio.to_thread.run_sync(
            lambda: WhisperModel(self.settings.whisper_model, device=device, compute_type=compute_type)
        )

    async def stop(self) -> None:
        self.model = None

    def create_stream(self, *, language: str | None):
        if self.model is None:
            raise RuntimeError("Whisper model is not loaded")
        return WhisperSTTStream(
            model=self.model,
            language=language,
            partial_interval_ms=self.settings.stt_partial_interval_ms,
            silence_finalize_ms=self.settings.stt_silence_finalize_ms,
            max_segment_seconds=self.settings.stt_max_segment_seconds,
        )


class WhisperSTTStream:
    def __init__(
        self,
        *,
        model,
        language: str | None,
        partial_interval_ms: int,
        silence_finalize_ms: int,
        max_segment_seconds: float,
    ):
        self.model = model
        self.language = language
        self.partial_interval_ms = partial_interval_ms
        self.max_segment_ms = max_segment_seconds * 1000.0
        self.vad = _StreamingVad(min_silence_duration_ms=silence_finalize_ms)
        self.active = False
        self.buffer = bytearray()
        self.elapsed_ms = 0.0
        self.segment_start_ms: int | None = None
        self.next_partial_ms = float(partial_interval_ms)
        self.last_partial_text = ""

    async def receive_audio(self, frame: bytes, *, sample_rate: int) -> list[TranscriptEvent]:
        if sample_rate != INPUT_SAMPLE_RATE:
            raise ValueError("Whisper STT expects 16 kHz pcm_s16le input")

        samples = pcm16_to_float32(frame)
        duration_ms = pcm16_duration_ms(frame, sample_rate=sample_rate)
        frame_start_ms = int(self.elapsed_ms)
        self.elapsed_ms += duration_ms
        vad_events = self.vad.observe(samples)

        events: list[TranscriptEvent] = []
        if "start" in vad_events and not self.active:
            self.active = True
            self.segment_start_ms = frame_start_ms
            events.append(TranscriptEvent(kind="speech_started", start_ms=frame_start_ms))

        if self.active:
            self.buffer.extend(frame)
            buffered_ms = len(self.buffer) // 2 / INPUT_SAMPLE_RATE * 1000.0
            if buffered_ms >= self.next_partial_ms:
                text = await self._transcribe(final=False)
                if text and text != self.last_partial_text:
                    self.last_partial_text = text
                    events.append(
                        TranscriptEvent(
                            kind="partial",
                            text=text,
                            start_ms=self.segment_start_ms,
                            end_ms=int(self.elapsed_ms),
                            language=self.language,
                        )
                    )
                self.next_partial_ms += self.partial_interval_ms

            if "end" in vad_events or buffered_ms >= self.max_segment_ms:
                events.extend(await self._finalize(end_ms=int(self.elapsed_ms)))

        return events

    async def commit(self) -> list[TranscriptEvent]:
        if not self.buffer and not self.active:
            return []
        return await self._finalize(end_ms=int(self.elapsed_ms))

    async def _finalize(self, *, end_ms: int) -> list[TranscriptEvent]:
        text = await self._transcribe(final=True)
        events = [TranscriptEvent(kind="speech_stopped", end_ms=end_ms)]
        if text:
            events.append(
                TranscriptEvent(
                    kind="final",
                    text=text,
                    start_ms=self.segment_start_ms,
                    end_ms=end_ms,
                    language=self.language,
                )
            )
        self.buffer.clear()
        self.active = False
        self.segment_start_ms = None
        self.next_partial_ms = float(self.partial_interval_ms)
        self.last_partial_text = ""
        self.vad.reset()
        return events

    async def _transcribe(self, *, final: bool) -> str:
        audio = pcm16_to_float32(bytes(self.buffer))

        def run() -> str:
            segments, _info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=final,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            return "".join(segment.text for segment in segments).strip()

        return await anyio.to_thread.run_sync(run)


class _StreamingVad:
    def __init__(self, *, min_silence_duration_ms: int):
        self.min_silence_duration_ms = min_silence_duration_ms
        self._impl = self._build_silero(min_silence_duration_ms)
        self._energy = _EnergyVad(min_silence_duration_ms=min_silence_duration_ms)

    def observe(self, samples) -> set[str]:
        if self._impl is None:
            return self._energy.observe(samples)
        return self._impl.observe(samples)

    def reset(self) -> None:
        if self._impl is not None:
            self._impl.reset()
        self._energy.reset()

    def _build_silero(self, min_silence_duration_ms: int):
        try:
            from silero_vad import VADIterator, load_silero_vad
            import torch
        except ImportError:
            return None
        model = load_silero_vad()
        iterator = VADIterator(
            model,
            sampling_rate=INPUT_SAMPLE_RATE,
            min_silence_duration_ms=min_silence_duration_ms,
        )
        return _SileroVad(iterator=iterator, torch=torch)


class _SileroVad:
    def __init__(self, *, iterator, torch):
        self.iterator = iterator
        self.torch = torch
        self.pending = None

    def observe(self, samples) -> set[str]:
        import numpy as np

        samples = np.asarray(samples, dtype=np.float32)
        if self.pending is not None and self.pending.size:
            samples = np.concatenate([self.pending, samples])

        events: set[str] = set()
        window = 512
        offset = 0
        while offset + window <= samples.size:
            chunk = samples[offset : offset + window]
            result = self.iterator(self.torch.from_numpy(chunk))
            if isinstance(result, dict):
                if "start" in result:
                    events.add("start")
                if "end" in result:
                    events.add("end")
            offset += window
        self.pending = samples[offset:].copy()
        return events

    def reset(self) -> None:
        self.pending = None
        self.iterator.reset_states()


class _EnergyVad:
    def __init__(self, *, min_silence_duration_ms: int):
        self.min_silence_duration_ms = min_silence_duration_ms
        self.active = False
        self.silence_ms = 0.0

    def observe(self, samples) -> set[str]:
        import numpy as np

        if samples.size == 0:
            return set()
        rms = float(np.sqrt(np.mean(np.square(samples))))
        duration_ms = samples.size / INPUT_SAMPLE_RATE * 1000.0
        events: set[str] = set()

        if rms >= 0.012:
            self.silence_ms = 0.0
            if not self.active:
                self.active = True
                events.add("start")
            return events

        if self.active:
            self.silence_ms += duration_ms
            if self.silence_ms >= self.min_silence_duration_ms:
                self.active = False
                self.silence_ms = 0.0
                events.add("end")
        return events

    def reset(self) -> None:
        self.active = False
        self.silence_ms = 0.0


def _resolve_device(configured: str) -> str:
    if configured != "auto":
        return configured
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_compute_type(configured: str, device: str) -> str:
    if configured != "auto":
        return configured
    return "float16" if device == "cuda" else "int8"
