# muzzle

Local streaming speech API for voice agents.

Muzzle runs a FastAPI service with:

- streaming TTS from Chatterbox Turbo
- streaming STT from Faster-Whisper
- WebSocket sessions for live mic input and generated audio output

## Quickstart

Install the TTS/STT dependencies:

```bash
make sync-real
make torch-check
```

Start the service:

```bash
make dev
```

The first startup downloads model weights for Chatterbox and Whisper. That can take a while.

If your GPU backend is not detected correctly, pick it explicitly:

```bash
TORCH_BACKEND=cu128 make sync-real
TORCH_BACKEND=cu130 make sync-real
```

`make dev` intentionally uses `uv run --no-sync`, so run `make sync-real` first. That avoids uv undoing the repaired torch install.

## Try TTS

With `make dev` running:

```bash
make example-tts
```

Use higher quality audio:

```bash
make example-tts QUALITY=high
```

Disable playback and just print events:

```bash
make example-tts TEXT="Hello from Muzzle." QUALITY=high PLAYER=none
```

Save raw PCM output:

```bash
make example-pcm PCM_OUT=/tmp/muzzle-example.pcm
```

Replay it with PipeWire:

```bash
pw-cat --playback --raw --format s16 --rate 24000 --channels 1 /tmp/muzzle-example.pcm
```

Play and benchmark streaming TTS directly on CPU, without starting the API server or STT model:

```bash
make example-tts-cpu
```

The CPU example uses the `cpu-smooth` quality profile, plays through `pw-cat` when PipeWire is available, prints the resolved `chunk_tokens` and `crossfade_ms`, prints per-chunk timing, and ends with a final `speed=` value. `speed=1.00x` means the model generated one second of audio per wall-clock second; values near or above `1.00x` are near real time. Use `CPU_QUALITY=fast`, `CPU_QUALITY=balanced`, or `CPU_QUALITY=high` to compare other profiles, `CROSSFADE_MS=40 CHUNK_TOKENS=64` to tune chunk joins manually, `LATENCY_MS=500` to test whether live playback is underrunning, `PLAYER=none` to disable playback, and `PCM_OUT=/tmp/muzzle-cpu.pcm` to save the streamed audio.

## Try STT

With `make dev` running:

```bash
make example-stt
```

Record longer:

```bash
make example-stt STT_SECONDS=20
```

## API

Health:

```text
GET /healthz
```

Voices:

```text
GET /v1/voices
POST /v1/voices
DELETE /v1/voices/{voice_id}
```

Sessions:

```text
GET /v1/sessions
```

`/v1/sessions` is a WebSocket endpoint. Send JSON control events and binary mic frames.

Example client events:

```json
{"type":"session.configure","voice_id":"default","stt_language":"en"}
{"type":"tts.speak","request_id":"tts-1","text":"Hello from Chatterbox Turbo.","quality":"high"}
{"type":"tts.cancel","request_id":"tts-1"}
{"type":"input_audio.commit"}
{"type":"session.close"}
```

Client mic audio is raw mono `pcm_s16le`, 16 kHz. Keep frames around 20-100 ms for normal interactive use.

TTS audio is returned as:

1. JSON `tts.audio.chunk` metadata
2. the matching binary `pcm_s16le` audio frame

Example server events:

```json
{"type":"stt.partial","text":"hello","is_final":false}
{"type":"stt.final","text":"hello world","is_final":true}
{"type":"tts.audio.chunk","request_id":"tts-1","index":0,"sample_rate":24000,"bytes":5760}
{"type":"tts.done","request_id":"tts-1","status":"completed"}
```

## TTS Quality

Set quality per generation:

```json
{"type":"tts.speak","request_id":"tts-1","text":"Hello.","quality":"high"}
```

Profiles:

| Quality | Best for | Tradeoff |
| --- | --- | --- |
| `fast` | lowest latency | rougher chunk joins |
| `balanced` | default interactive use | moderate latency |
| `high` | smoother audio | higher latency |
| `cpu-smooth` | CPU streaming tests | smoother joins with larger chunks |

You can still override generation knobs directly:

```json
{"type":"tts.speak","request_id":"tts-1","text":"Hello.","quality":"high","chunk_tokens":80,"crossfade_ms":40}
```

## Voice Uploads

Create a managed voice with multipart form data:

- `name`: display name
- `reference_audio`: reference clip

For Chatterbox, use clean reference audio with one speaker and at least 5 seconds. Ten to thirty seconds is usually better.

Prepared voices are stored under:

```text
data/voices/{voice_id}
```

The built-in `default` voice cannot be deleted.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MUZZLE_MODEL_BACKEND` | `real` | `real` or `fake` |
| `MUZZLE_DATA_DIR` | `data` | Voice metadata, reference audio, and cached conditionals |
| `MUZZLE_AUTH_TOKEN` | unset | Optional bearer token for HTTP and WebSocket requests |
| `MUZZLE_TTS_DEVICE` | `auto` | `cuda`, `mps`, `cpu`, or `auto` |
| `MUZZLE_WHISPER_MODEL` | `distil-large-v3` | Faster-Whisper model name/path |
| `MUZZLE_WHISPER_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |

## Docker

Run tests in Docker:

```bash
make docker-test
```

Run the service in Docker:

```bash
make docker-run
```

Run CPU-only inference in Docker:

```bash
make docker-build-cpu
make docker-run-cpu
```

Build a pinned CUDA image instead:

```bash
make docker-build-cu128
make docker-build-cu130
```

Run one with GPU access:

```bash
make docker-run-cu128
make docker-run-cu130
```

The pinned images use `Dockerfile.cu128` and `Dockerfile.cu130`.
They are based on NVIDIA CUDA cuDNN runtime images, and PyTorch is installed without its vendored CUDA wheel dependencies.
The CPU image uses the shared `Dockerfile` with the PyTorch CPU wheel index and forces TTS/STT devices to `cpu` at runtime.

Local Docker tags are `muzzle:fake`, `muzzle:cpu`, `muzzle:cu128`, and `muzzle:cu130`. `muzzle:latest` is an alias for the CPU image.
Docker builds install PyTorch from explicit PyTorch wheel indexes and pin matching `torch` and `torchaudio` versions with `DOCKER_TORCH_VERSION`, which defaults to `2.11.0`.

## Testing

Run the API tests:

```bash
make test
```

The tests use the fake backend, so they do not download model weights.

You can also run the fake server directly while working on API plumbing:

```bash
make sync
make dev-fake
```

## Development

Useful targets:

```bash
make sync
make sync-real
make test
make dev-fake
make dev
make example-tts
make example-stt
make torch-check
```

TTS uses the pinned streaming Chatterbox fork:

```text
github.com/chrishayen/chatterbox @ ce5a900dbdbc776eeb3d00c1c1607143c7604a5b
```
