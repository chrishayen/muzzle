from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

SUPPORTED_INPUT_FORMAT = "pcm_s16le"
SUPPORTED_OUTPUT_FORMAT = "pcm_s16le"
INPUT_SAMPLE_RATE = 16_000


class SessionConfigureEvent(BaseModel):
    type: Literal["session.configure"]
    input_audio_format: Literal["pcm_s16le"] = SUPPORTED_INPUT_FORMAT
    input_sample_rate: Literal[16000] = INPUT_SAMPLE_RATE
    output_audio_format: Literal["pcm_s16le"] = SUPPORTED_OUTPUT_FORMAT
    voice_id: str = "default"
    stt_language: str | None = "en"


class TTSSpeakEvent(BaseModel):
    type: Literal["tts.speak"]
    request_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=8000)
    voice_id: str | None = None
    chunk_tokens: int = Field(default=24, ge=1, le=128)
    crossfade_ms: float = Field(default=12.0, ge=0, le=100)
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    top_k: int = Field(default=1000, ge=1, le=5000)
    repetition_penalty: float = Field(default=1.2, ge=0.1, le=10)
    max_gen_len: int = Field(default=1000, ge=1, le=4000)


class TTSCancelEvent(BaseModel):
    type: Literal["tts.cancel"]
    request_id: str | None = None


class InputAudioCommitEvent(BaseModel):
    type: Literal["input_audio.commit"]


class SessionCloseEvent(BaseModel):
    type: Literal["session.close"]


ClientEvent = Annotated[
    Union[
        SessionConfigureEvent,
        TTSSpeakEvent,
        TTSCancelEvent,
        InputAudioCommitEvent,
        SessionCloseEvent,
    ],
    Field(discriminator="type"),
]
CLIENT_EVENT_ADAPTER = TypeAdapter(ClientEvent)


class SessionCreatedEvent(BaseModel):
    type: Literal["session.created"] = "session.created"
    session_id: str
    input_audio_format: str = SUPPORTED_INPUT_FORMAT
    input_sample_rate: int = INPUT_SAMPLE_RATE
    output_audio_format: str = SUPPORTED_OUTPUT_FORMAT


class SessionUpdatedEvent(BaseModel):
    type: Literal["session.updated"] = "session.updated"
    voice_id: str
    stt_language: str | None


class SpeechStartedEvent(BaseModel):
    type: Literal["input_audio.speech_started"] = "input_audio.speech_started"
    start_ms: int | None = None


class SpeechStoppedEvent(BaseModel):
    type: Literal["input_audio.speech_stopped"] = "input_audio.speech_stopped"
    end_ms: int | None = None


class STTPartialEvent(BaseModel):
    type: Literal["stt.partial"] = "stt.partial"
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    language: str | None = None
    is_final: Literal[False] = False


class STTFinalEvent(BaseModel):
    type: Literal["stt.final"] = "stt.final"
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    language: str | None = None
    is_final: Literal[True] = True


class TTSStartedEvent(BaseModel):
    type: Literal["tts.started"] = "tts.started"
    request_id: str
    voice_id: str


class TTSAudioChunkEvent(BaseModel):
    type: Literal["tts.audio.chunk"] = "tts.audio.chunk"
    request_id: str
    index: int
    sample_rate: int
    start_sample: int
    end_sample: int
    is_final: bool
    bytes: int
    generated_tokens: int
    watermarked: bool


class TTSDoneEvent(BaseModel):
    type: Literal["tts.done"] = "tts.done"
    request_id: str
    status: Literal["completed", "cancelled", "error"]


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
    request_id: str | None = None
    fatal: bool = False


def parse_client_event(payload: str):
    try:
        return CLIENT_EVENT_ADAPTER.validate_json(payload)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def event_payload(event: BaseModel) -> dict:
    return event.model_dump(mode="json", exclude_none=True)
