from __future__ import annotations

import io
import struct
import wave

from fastapi.testclient import TestClient

from muzzle.app import create_app
from muzzle.config import Settings


def _client(tmp_path, **overrides):
    settings = Settings(
        data_dir=tmp_path,
        model_backend="fake",
        stt_partial_interval_ms=40,
        stt_silence_finalize_ms=40,
        **overrides,
    )
    return TestClient(create_app(settings))


def _wav_bytes(duration_seconds: float, sample_rate: int = 16_000) -> bytes:
    buf = io.BytesIO()
    frames = b"\x00\x00" * int(duration_seconds * sample_rate)
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return buf.getvalue()


def _speech_frame(duration_ms: int = 100, sample_rate: int = 16_000) -> bytes:
    sample_count = int(sample_rate * duration_ms / 1000)
    return b"".join(struct.pack("<h", 8000) for _ in range(sample_count))


def _silence_frame(duration_ms: int = 100, sample_rate: int = 16_000) -> bytes:
    return b"\x00\x00" * int(sample_rate * duration_ms / 1000)


def _collect_tts_chunks(ws, message: dict) -> list[dict]:
    ws.send_json(message)
    started = ws.receive_json()
    assert started["type"] == "tts.started"
    assert started["request_id"] == message["request_id"]

    chunks = []
    while True:
        event = ws.receive_json()
        if event["type"] == "tts.done":
            assert event == {"type": "tts.done", "request_id": message["request_id"], "status": "completed"}
            return chunks
        assert event["type"] == "tts.audio.chunk"
        chunks.append(event)
        assert len(ws.receive_bytes()) == event["bytes"]


def test_health_and_default_voice(tmp_path):
    with _client(tmp_path) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["ok"] is True
        assert response.json()["model_backend"] == "fake"

        voices = client.get("/v1/voices").json()["voices"]
        assert voices == [
            {
                "voice_id": "default",
                "name": "Fake default",
                "source": "builtin",
                "created_at": None,
                "sample_rate": None,
                "duration_seconds": None,
            }
        ]


def test_voice_upload_list_and_delete(tmp_path):
    with _client(tmp_path) as client:
        response = client.post(
            "/v1/voices",
            data={"name": "Agent voice"},
            files={"reference_audio": ("voice.wav", _wav_bytes(6.0), "audio/wav")},
        )
        assert response.status_code == 201
        voice = response.json()["voice"]
        assert voice["name"] == "Agent voice"
        assert voice["source"] == "managed"
        assert voice["sample_rate"] == 16000
        assert voice["duration_seconds"] == 6.0

        voices = client.get("/v1/voices").json()["voices"]
        assert {item["voice_id"] for item in voices} == {"default", voice["voice_id"]}

        delete_response = client.delete(f"/v1/voices/{voice['voice_id']}")
        assert delete_response.status_code == 200
        assert delete_response.json() == {"deleted": True}


def test_short_voice_upload_is_rejected(tmp_path):
    with _client(tmp_path) as client:
        response = client.post(
            "/v1/voices",
            data={"name": "Too short"},
            files={"reference_audio": ("voice.wav", _wav_bytes(1.0), "audio/wav")},
        )
        assert response.status_code == 400
        assert "at least 5 seconds" in response.json()["detail"]


def test_websocket_stt_tts_and_binary_ordering(tmp_path):
    with _client(tmp_path) as client:
        with client.websocket_connect("/v1/sessions") as ws:
            created = ws.receive_json()
            assert created["type"] == "session.created"
            assert created["input_sample_rate"] == 16000

            ws.send_bytes(_speech_frame())
            assert ws.receive_json()["type"] == "input_audio.speech_started"
            partial = ws.receive_json()
            assert partial["type"] == "stt.partial"
            assert partial["is_final"] is False

            ws.send_bytes(_silence_frame())
            assert ws.receive_json()["type"] == "input_audio.speech_stopped"
            final = ws.receive_json()
            assert final["type"] == "stt.final"
            assert final["is_final"] is True

            ws.send_json({"type": "tts.speak", "request_id": "tts-1", "text": "hello"})
            assert ws.receive_json() == {
                "type": "tts.started",
                "request_id": "tts-1",
                "voice_id": "default",
            }

            first_chunk = ws.receive_json()
            assert first_chunk["type"] == "tts.audio.chunk"
            assert first_chunk["request_id"] == "tts-1"
            assert first_chunk["index"] == 0
            first_audio = ws.receive_bytes()
            assert len(first_audio) == first_chunk["bytes"]

            second_chunk = ws.receive_json()
            assert second_chunk["type"] == "tts.audio.chunk"
            assert second_chunk["index"] == 1
            second_audio = ws.receive_bytes()
            assert len(second_audio) == second_chunk["bytes"]

            done = ws.receive_json()
            assert done == {"type": "tts.done", "request_id": "tts-1", "status": "completed"}


def test_websocket_bad_audio_error(tmp_path):
    with _client(tmp_path) as client:
        with client.websocket_connect("/v1/sessions") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_bytes(b"\x00")
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "bad_audio"


def test_tts_quality_profile_and_explicit_overrides(tmp_path):
    with _client(tmp_path) as client:
        with client.websocket_connect("/v1/sessions") as ws:
            assert ws.receive_json()["type"] == "session.created"

            high_chunks = _collect_tts_chunks(
                ws,
                {"type": "tts.speak", "request_id": "tts-high", "text": "hello", "quality": "high"},
            )
            assert high_chunks[0]["generated_tokens"] == 64

            cpu_smooth_chunks = _collect_tts_chunks(
                ws,
                {
                    "type": "tts.speak",
                    "request_id": "tts-cpu-smooth",
                    "text": "hello",
                    "quality": "cpu-smooth",
                },
            )
            assert cpu_smooth_chunks[0]["generated_tokens"] == 64

            override_chunks = _collect_tts_chunks(
                ws,
                {
                    "type": "tts.speak",
                    "request_id": "tts-override",
                    "text": "hello",
                    "quality": "high",
                    "chunk_tokens": 12,
                },
            )
            assert override_chunks[0]["generated_tokens"] == 12
