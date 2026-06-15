# muzzle

Local streaming speech API for full-duplex voice sessions.

## What is implemented

- FastAPI service with `GET /healthz`, managed voice endpoints, and `GET /v1/sessions` WebSocket sessions.
- Raw mono PCM16 streaming:
  - client mic input: `pcm_s16le`, 16 kHz, mono
  - TTS output: `pcm_s16le`, sample rate reported per chunk
- Speech-only session API:
  - binary client frames are treated as mic audio for STT
  - JSON client events control session config, TTS, cancellation, and audio commit
  - server sends STT events, TTS metadata events, and TTS binary audio chunks
- Real adapters:
  - TTS: Chatterbox Turbo streaming fork pinned to `ce5a900dbdbc776eeb3d00c1c1607143c7604a5b`
  - STT: `faster-whisper`, default model `distil-large-v3`
  - VAD: Silero streaming VAD when installed, with an energy fallback
- Fake adapters for fast local API tests without downloading model weights.

## Install

```bash
make sync
```

For the real service:

```bash
make sync-real
make torch-check
```

`make sync-real` uses uv's PyTorch backend support after installing the real dependencies:

```bash
UV_TORCH_BACKEND=auto uv pip install --torch-backend auto --upgrade --reinstall torch torchaudio
```

This is required on newer GPUs such as RTX 50-series cards because the Chatterbox dependency pin can otherwise install an older CUDA wheel that lacks `sm_120` kernels. The Makefile defaults to `TORCH_BACKEND=auto`; if auto cannot inspect the GPU/driver, pick an explicit backend:

```bash
TORCH_BACKEND=cu128 make sync-real
TORCH_BACKEND=cu130 make sync-real
```

## Run The Real Service

After `make sync-real`, start the real service:

```bash
make dev
```

This loads Chatterbox Turbo and Faster-Whisper. The first run downloads model weights through the underlying libraries:

- Chatterbox: `ChatterboxTurboTTS.from_pretrained()` uses Hugging Face `snapshot_download()`.
- STT: `faster_whisper.WhisperModel()` resolves/downloads the configured Whisper model.

Make targets:

```bash
make sync-real
make torch-reinstall
make torch-check
make test
make dev
make example-tts
make example-stt
make example-pcm
make docker-test
make docker-run
```

Real run/example targets use `uv run --no-sync` so uv does not immediately restore the transitive Chatterbox torch pin after `make sync-real` repairs the local environment.

`make test`, `make docker-test`, and `make dev-fake` use a fake backend for API contract tests only. They do not test real TTS or STT.

To play real TTS through your Hyprland desktop audio output, start the real server and run the example client in another terminal:

```bash
make dev
make example-tts
```

The TTS example uses `pw-cat` when available, which routes audio through PipeWire to the default desktop sink. Override the prompt or disable playback for a protocol smoke test:

```bash
make example-tts TEXT="Testing the streaming path." PLAYER=none
```

To test real STT from your microphone:

```bash
make example-stt
```

Speak for the configured duration, then the client sends `input_audio.commit` and prints transcript events. Override the capture length:

```bash
make example-stt STT_SECONDS=20
```

`make example-pcm` writes raw real TTS `pcm_s16le` to `/tmp/muzzle-example.pcm` by default. Replay it manually through PipeWire:

```bash
pw-cat --playback --raw --format s16 --rate 24000 --channels 1 /tmp/muzzle-example.pcm
```

Docker real builds also accept the same backend selector:

```bash
TORCH_BACKEND=cu128 make docker-build-real
make docker-run
```

Use an explicit `TORCH_BACKEND` for Docker when the build environment cannot access the NVIDIA driver for uv's `auto` detection.

Useful environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MUZZLE_MODEL_BACKEND` | `real` | `real` or `fake` |
| `MUZZLE_DATA_DIR` | `data` | Local voice metadata, reference audio, and cached conditionals |
| `MUZZLE_AUTH_TOKEN` | unset | Optional bearer token for HTTP and WebSocket requests |
| `MUZZLE_TTS_DEVICE` | `auto` | `cuda`, `mps`, `cpu`, or `auto` |
| `MUZZLE_WHISPER_MODEL` | `distil-large-v3` | Faster-Whisper model name/path |
| `MUZZLE_WHISPER_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |

## HTTP API

`GET /healthz`

Returns service readiness, backend, and session counts.

`GET /v1/voices`

Lists the built-in default voice, when available, plus managed voices.

`POST /v1/voices`

Multipart form:

- `name`: display name
- `reference_audio`: reference clip; Chatterbox requires more than 5 seconds

The real backend prepares and stores Chatterbox conditionals under `data/voices/{voice_id}`.

`DELETE /v1/voices/{voice_id}`

Deletes a managed voice. The `default` voice cannot be deleted.

## WebSocket API

Connect to `GET /v1/sessions`.

Client JSON events:

```json
{"type":"session.configure","voice_id":"default","stt_language":"en"}
{"type":"tts.speak","request_id":"tts-1","text":"Hello from Chatterbox Turbo."}
{"type":"tts.cancel","request_id":"tts-1"}
{"type":"input_audio.commit"}
{"type":"session.close"}
```

Client binary frames are raw 16 kHz mono signed 16-bit little-endian PCM. Keep frames around 20-100 ms for normal interactive use.

TTS output order is deliberate:

1. JSON `tts.audio.chunk` metadata
2. the matching binary PCM16 audio frame

Example server events:

```json
{"type":"stt.partial","text":"hello","is_final":false}
{"type":"stt.final","text":"hello world","is_final":true}
{"type":"tts.audio.chunk","request_id":"tts-1","index":0,"sample_rate":24000,"bytes":5760}
{"type":"tts.done","request_id":"tts-1","status":"completed"}
```

## Tests

```bash
uv run --extra test pytest -q
```
