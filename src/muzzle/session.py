from __future__ import annotations

import asyncio
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from .adapters.base import STTAdapter, TTSAdapter
from .audio import INPUT_SAMPLE_RATE, AudioFormatError, validate_pcm16_frame
from .config import Settings
from .domain import TTSOptions, TranscriptEvent
from .schemas import (
    ErrorEvent,
    InputAudioCommitEvent,
    SessionCloseEvent,
    SessionConfigureEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    STTFinalEvent,
    STTPartialEvent,
    TTSAudioChunkEvent,
    TTSCancelEvent,
    TTSDoneEvent,
    TTSSpeakEvent,
    TTSStartedEvent,
    event_payload,
    parse_client_event,
    SpeechStartedEvent,
    SpeechStoppedEvent,
)
from .voice_registry import VoiceRegistry


class VoiceSession:
    def __init__(
        self,
        *,
        websocket: WebSocket,
        settings: Settings,
        tts_adapter: TTSAdapter,
        stt_adapter: STTAdapter,
        voice_registry: VoiceRegistry,
    ):
        self.websocket = websocket
        self.settings = settings
        self.tts_adapter = tts_adapter
        self.voice_registry = voice_registry
        self.session_id = uuid.uuid4().hex
        self.voice_id = "default"
        self.stt_language = settings.stt_language
        self.stt_stream = stt_adapter.create_stream(language=self.stt_language)
        self.send_lock = asyncio.Lock()
        self.tts_tasks: dict[str, asyncio.Task] = {}
        self.closed = False

    async def run(self) -> None:
        await self._send_event(SessionCreatedEvent(session_id=self.session_id))
        try:
            while not self.closed:
                message = await self.websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    await self._handle_text(message["text"])
                elif message.get("bytes") is not None:
                    await self._handle_audio(message["bytes"])
        except WebSocketDisconnect:
            pass
        finally:
            self.closed = True
            await self._cancel_tts()

    async def _handle_text(self, payload: str) -> None:
        try:
            event = parse_client_event(payload)
        except ValueError as exc:
            await self._send_error("bad_event", str(exc))
            return

        if isinstance(event, SessionConfigureEvent):
            self.voice_id = event.voice_id
            self.stt_language = event.stt_language
            self.stt_stream = self.stt_adapter.create_stream(language=self.stt_language)
            await self._send_event(SessionUpdatedEvent(voice_id=self.voice_id, stt_language=self.stt_language))
            return

        if isinstance(event, TTSSpeakEvent):
            request_id = event.request_id
            if request_id in self.tts_tasks:
                await self._send_error("duplicate_request", f"request_id {request_id!r} is already active", request_id)
                return
            task = asyncio.create_task(self._run_tts(event), name=f"tts:{request_id}")
            self.tts_tasks[request_id] = task
            task.add_done_callback(lambda _task, rid=request_id: self.tts_tasks.pop(rid, None))
            return

        if isinstance(event, TTSCancelEvent):
            await self._cancel_tts(event.request_id)
            return

        if isinstance(event, InputAudioCommitEvent):
            await self._emit_transcript_events(await self.stt_stream.commit())
            return

        if isinstance(event, SessionCloseEvent):
            self.closed = True
            await self.websocket.close(code=1000)

    async def _handle_audio(self, frame: bytes) -> None:
        try:
            validate_pcm16_frame(frame, sample_rate=INPUT_SAMPLE_RATE)
            events = await self.stt_stream.receive_audio(frame, sample_rate=INPUT_SAMPLE_RATE)
        except (AudioFormatError, ValueError) as exc:
            await self._send_error("bad_audio", str(exc))
            return
        await self._emit_transcript_events(events)

    async def _run_tts(self, event: TTSSpeakEvent) -> None:
        voice_id = event.voice_id or self.voice_id
        voice = None
        if voice_id != "default":
            voice = self.voice_registry.get(voice_id)
            if voice is None:
                await self._send_error("voice_not_found", f"unknown voice_id {voice_id!r}", event.request_id)
                await self._send_event(TTSDoneEvent(request_id=event.request_id, status="error"))
                return

        options = TTSOptions(
            request_id=event.request_id,
            text=event.text,
            voice_id=voice_id,
            chunk_tokens=event.chunk_tokens,
            crossfade_ms=event.crossfade_ms,
            temperature=event.temperature,
            top_p=event.top_p,
            top_k=event.top_k,
            repetition_penalty=event.repetition_penalty,
            max_gen_len=event.max_gen_len,
        )

        try:
            await self._send_event(TTSStartedEvent(request_id=event.request_id, voice_id=voice_id))
            async for chunk in self.tts_adapter.synthesize_stream(options, voice):
                await self._send_tts_chunk(chunk)
            await self._send_event(TTSDoneEvent(request_id=event.request_id, status="completed"))
        except asyncio.CancelledError:
            await self._send_event(TTSDoneEvent(request_id=event.request_id, status="cancelled"))
        except Exception as exc:
            await self._send_error("tts_failed", str(exc), event.request_id)
            await self._send_event(TTSDoneEvent(request_id=event.request_id, status="error"))

    async def _emit_transcript_events(self, events: list[TranscriptEvent]) -> None:
        for event in events:
            if event.kind == "speech_started":
                await self._send_event(SpeechStartedEvent(start_ms=event.start_ms))
            elif event.kind == "speech_stopped":
                await self._send_event(SpeechStoppedEvent(end_ms=event.end_ms))
            elif event.kind == "partial":
                await self._send_event(
                    STTPartialEvent(
                        text=event.text,
                        start_ms=event.start_ms,
                        end_ms=event.end_ms,
                        language=event.language,
                    )
                )
            elif event.kind == "final":
                await self._send_event(
                    STTFinalEvent(
                        text=event.text,
                        start_ms=event.start_ms,
                        end_ms=event.end_ms,
                        language=event.language,
                    )
                )

    async def _cancel_tts(self, request_id: str | None = None) -> None:
        tasks = list(self.tts_tasks.items())
        for rid, task in tasks:
            if request_id is None or request_id == rid:
                task.cancel()
        await asyncio.gather(
            *(task for rid, task in tasks if request_id is None or request_id == rid),
            return_exceptions=True,
        )

    async def _send_error(self, code: str, message: str, request_id: str | None = None) -> None:
        await self._send_event(ErrorEvent(code=code, message=message, request_id=request_id))

    async def _send_event(self, event) -> None:
        if self.websocket.client_state == WebSocketState.DISCONNECTED:
            return
        async with self.send_lock:
            await self.websocket.send_json(event_payload(event))

    async def _send_tts_chunk(self, chunk) -> None:
        if self.websocket.client_state == WebSocketState.DISCONNECTED:
            return
        event = TTSAudioChunkEvent(
            request_id=chunk.request_id,
            index=chunk.index,
            sample_rate=chunk.sample_rate,
            start_sample=chunk.start_sample,
            end_sample=chunk.end_sample,
            is_final=chunk.is_final,
            bytes=len(chunk.audio),
            generated_tokens=chunk.generated_tokens,
            watermarked=chunk.watermarked,
        )
        async with self.send_lock:
            await self.websocket.send_json(event_payload(event))
            await self.websocket.send_bytes(chunk.audio)
