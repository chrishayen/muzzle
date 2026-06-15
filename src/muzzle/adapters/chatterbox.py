from __future__ import annotations

import asyncio

import anyio

from ..config import Settings
from ..domain import TTSChunk, TTSOptions, VoiceInfo, VoiceRecord


class ChatterboxTTSAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self._audio_to_pcm_s16le = None
        self._conditionals_cls = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        try:
            from chatterbox.streaming import audio_to_pcm_s16le
            from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals
        except ModuleNotFoundError as exc:
            if exc.name == "chatterbox" or exc.name.startswith("chatterbox."):
                raise RuntimeError("Chatterbox is not installed. Run `make sync-real` before `make dev`.") from exc
            raise

        device = await anyio.to_thread.run_sync(_resolve_torch_device, self.settings.tts_device)
        self.model = await anyio.to_thread.run_sync(ChatterboxTurboTTS.from_pretrained, device)
        self._audio_to_pcm_s16le = audio_to_pcm_s16le
        self._conditionals_cls = Conditionals

    async def stop(self) -> None:
        self.model = None

    def default_voice(self) -> VoiceInfo | None:
        if self.model is None or getattr(self.model, "conds", None) is None:
            return None
        return VoiceInfo(voice_id="default", name="Chatterbox Turbo default", source="builtin")

    async def prepare_voice(self, record: VoiceRecord) -> None:
        self._ensure_started()

        def prepare() -> None:
            self.model.prepare_conditionals(str(record.reference_audio_path))
            self.model.conds.save(record.directory / "conds.pt")

        async with self._lock:
            await anyio.to_thread.run_sync(prepare)

    async def synthesize_stream(self, options: TTSOptions, voice: VoiceRecord | None):
        self._ensure_started()
        async with self._lock:
            if voice is not None:
                self._load_voice_conditionals(voice)

            stream = self.model.stream(
                options.text,
                chunk_tokens=options.chunk_tokens,
                crossfade_ms=options.crossfade_ms,
                temperature=options.temperature,
                top_p=options.top_p,
                top_k=options.top_k,
                repetition_penalty=options.repetition_penalty,
                max_gen_len=options.max_gen_len,
            )

            while True:
                chunk = await anyio.to_thread.run_sync(lambda: next(stream, None))
                if chunk is None:
                    break
                yield TTSChunk(
                    request_id=options.request_id,
                    audio=self._audio_to_pcm_s16le(chunk.audio),
                    sample_rate=chunk.sample_rate,
                    index=chunk.index,
                    is_final=chunk.is_final,
                    start_sample=chunk.start_sample,
                    end_sample=chunk.end_sample,
                    generated_tokens=chunk.generated_tokens,
                    watermarked=chunk.watermarked,
                )

    def _load_voice_conditionals(self, voice: VoiceRecord) -> None:
        conds_path = voice.directory / "conds.pt"
        if not conds_path.exists():
            raise ValueError(f"voice {voice.voice_id!r} has not been prepared")
        self.model.conds = self._conditionals_cls.load(conds_path, map_location="cpu").to(self.model.device)

    def _ensure_started(self) -> None:
        if self.model is None:
            raise RuntimeError("Chatterbox model is not loaded")


def _resolve_torch_device(configured: str) -> str:
    if configured != "auto":
        return configured

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
