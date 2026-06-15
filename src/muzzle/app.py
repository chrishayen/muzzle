from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket

from .adapters import build_stt_adapter, build_tts_adapter
from .adapters.base import STTAdapter, TTSAdapter
from .config import Settings
from .domain import VoiceInfo
from .session import VoiceSession
from .voice_registry import VoiceRegistry


class ServiceState:
    def __init__(
        self,
        *,
        settings: Settings,
        tts_adapter: TTSAdapter | None = None,
        stt_adapter: STTAdapter | None = None,
    ):
        self.settings = settings
        self.voice_registry = VoiceRegistry(settings.data_dir / "voices")
        self.tts_adapter = tts_adapter or build_tts_adapter(settings)
        self.stt_adapter = stt_adapter or build_stt_adapter(settings)
        self.active_sessions = 0
        self.session_lock = asyncio.Lock()
        self.ready = False

    async def start(self) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.voice_registry.ensure()
        await self.tts_adapter.start()
        await self.stt_adapter.start()
        self.ready = True

    async def stop(self) -> None:
        self.ready = False
        await self.tts_adapter.stop()
        await self.stt_adapter.stop()


def create_app(
    settings: Settings | None = None,
    *,
    tts_adapter: TTSAdapter | None = None,
    stt_adapter: STTAdapter | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    state = ServiceState(settings=settings, tts_adapter=tts_adapter, stt_adapter=stt_adapter)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await state.start()
        app.state.service = state
        try:
            yield
        finally:
            await state.stop()

    app = FastAPI(title="muzzle", version="0.1.0", lifespan=lifespan)

    async def require_auth(
        authorization: str | None = Header(default=None),
    ) -> None:
        if not settings.auth_token:
            return
        if authorization != f"Bearer {settings.auth_token}":
            raise HTTPException(status_code=401, detail="invalid bearer token")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "ok": state.ready,
            "model_backend": settings.model_backend,
            "active_sessions": state.active_sessions,
            "max_sessions": settings.max_sessions,
        }

    @app.get("/v1/voices", dependencies=[Depends(require_auth)])
    async def list_voices() -> dict:
        voices: list[VoiceInfo] = []
        default_voice = state.tts_adapter.default_voice()
        if default_voice is not None:
            voices.append(default_voice)
        voices.extend(record.to_info() for record in state.voice_registry.list_records())
        return {"voices": [asdict(voice) for voice in voices]}

    @app.post("/v1/voices", status_code=201, dependencies=[Depends(require_auth)])
    async def create_voice(
        name: str = Form(min_length=1, max_length=128),
        reference_audio: UploadFile = File(),
    ) -> dict:
        max_bytes = settings.max_upload_mb * 1024 * 1024
        content = await reference_audio.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail="reference audio is too large")

        record = state.voice_registry.create(
            name=name,
            filename=reference_audio.filename or "reference.wav",
            content=content,
        )
        try:
            await state.tts_adapter.prepare_voice(record)
        except Exception as exc:
            state.voice_registry.delete(record.voice_id)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"voice": asdict(record.to_info())}

    @app.delete("/v1/voices/{voice_id}", dependencies=[Depends(require_auth)])
    async def delete_voice(voice_id: str) -> dict:
        if voice_id == "default":
            raise HTTPException(status_code=400, detail="default voice cannot be deleted")
        deleted = state.voice_registry.delete(voice_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="voice not found")
        return {"deleted": True}

    @app.websocket("/v1/sessions")
    async def sessions(websocket: WebSocket) -> None:
        if settings.auth_token and websocket.headers.get("authorization") != f"Bearer {settings.auth_token}":
            await websocket.close(code=1008)
            return

        await websocket.accept()
        async with state.session_lock:
            if state.active_sessions >= settings.max_sessions:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "overloaded",
                        "message": "maximum active sessions reached",
                        "fatal": True,
                    }
                )
                await websocket.close(code=1013)
                return
            state.active_sessions += 1

        try:
            session = VoiceSession(
                websocket=websocket,
                settings=settings,
                tts_adapter=state.tts_adapter,
                stt_adapter=state.stt_adapter,
                voice_registry=state.voice_registry,
            )
            await session.run()
        finally:
            async with state.session_lock:
                state.active_sessions -= 1

    return app
