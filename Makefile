IMAGE ?= muzzle
DOCKER_FAKE_IMAGE ?= $(IMAGE):fake
DOCKER_CPU_IMAGE ?= $(IMAGE):cpu
DOCKER_LATEST_IMAGE ?= $(IMAGE):latest
DOCKER_CU128_IMAGE ?= $(IMAGE):cu128
DOCKER_CU130_IMAGE ?= $(IMAGE):cu130
PORT ?= 8000
UV ?= uv
TORCH_BACKEND ?= auto
DOCKER_TORCH_VERSION ?= 2.11.0
DOCKER_TORCHAUDIO_VERSION ?= $(DOCKER_TORCH_VERSION)
DOCKER_CU128_BASE_IMAGE ?= nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
DOCKER_CU130_BASE_IMAGE ?= nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04
API_URL ?= ws://127.0.0.1:$(PORT)/v1/sessions
TEXT ?= Hello from Muzzle, streaming text to speech in small chunks.
QUALITY ?= balanced
CPU_QUALITY ?= cpu-smooth
PLAYER ?= auto
LATENCY_MS ?= 120
CHUNK_TOKENS ?=
CROSSFADE_MS ?=
PCM_OUT ?=
PCM_ARGS = $(if $(PCM_OUT),--save-pcm "$(PCM_OUT)",)
TTS_KNOB_ARGS = $(if $(CHUNK_TOKENS),--chunk-tokens "$(CHUNK_TOKENS)",) $(if $(CROSSFADE_MS),--crossfade-ms "$(CROSSFADE_MS)",)
STT_SECONDS ?= 10
STT_INPUT ?=
STT_ARGS = $(if $(STT_INPUT),--input-pcm "$(STT_INPUT)",--duration-seconds "$(STT_SECONDS)")

.PHONY: help sync sync-real check-real-deps torch-reinstall torch-check test dev dev-real dev-fake example-tts example-tts-cpu example-stt example-pcm docker-build docker-build-real docker-build-cpu docker-build-cu128 docker-build-cu130 docker-test docker-run docker-run-fake docker-run-real docker-run-cpu docker-run-cu128 docker-run-cu130

help:
	@printf '%s\n' \
		'Targets:' \
		'  make sync              Install local test dependencies with uv' \
		'  make sync-real         Install real Chatterbox/Whisper deps and repair torch backend' \
		'  make torch-reinstall   Reinstall torch/torchaudio with uv torch backend selection' \
		'  make torch-check       Print the installed torch/CUDA/GPU details' \
		'  make test              Run fake-backend API contract tests' \
		'  make dev               Run the real API service locally after sync-real' \
		'  make dev-fake          Run fake API plumbing service locally' \
		'  make example-tts       Call real /v1/sessions TTS and play through PipeWire' \
		'  make example-tts-cpu   Play streaming TTS directly from CPU and print speed metrics' \
		'  make example-stt       Record mic with PipeWire and stream it to real STT' \
		'  make example-pcm       Save streamed real TTS as raw pcm_s16le for replay' \
		'  make docker-test       Build and run fake-backend tests in Docker' \
		'  make docker-run        Run CPU real API service in Docker' \
		'  make docker-run-fake   Run fake API plumbing service in Docker' \
		'  make docker-build-cpu   Build CPU-only real image' \
		'  make docker-run-cpu     Run CPU-only real image' \
		'  make docker-build-cu128  Build pinned CUDA 12.8 real image' \
		'  make docker-build-cu130  Build pinned CUDA 13.0 real image' \
		'  make docker-run-cu128    Run pinned CUDA 12.8 real image' \
		'  make docker-run-cu130    Run pinned CUDA 13.0 real image' \
		'' \
		'Variables:' \
		'  IMAGE=muzzle             Docker repository name; variant is the tag' \
		'  TORCH_BACKEND=auto       uv torch backend for local real installs; override with cu128/cu130/etc' \
		'  DOCKER_TORCH_VERSION=2.11.0 Docker torch/torchaudio version' \
		'  DOCKER_CU128_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04' \
		'  DOCKER_CU130_BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04'

sync:
	$(UV) sync --extra test

sync-real:
	UV_TORCH_BACKEND=$(TORCH_BACKEND) $(UV) sync --extra real --extra test
	$(MAKE) torch-reinstall

check-real-deps:
	@$(UV) run --no-sync python -c "import importlib.util, sys; modules = ('chatterbox', 'faster_whisper', 'torch', 'torchaudio'); missing = [m for m in modules if importlib.util.find_spec(m) is None]; sys.stderr.write('Missing real dependencies: ' + ', '.join(missing) + '\nRun: make sync-real\n') if missing else None; raise SystemExit(bool(missing))"

torch-reinstall:
	UV_TORCH_BACKEND=$(TORCH_BACKEND) $(UV) pip install --torch-backend "$(TORCH_BACKEND)" --upgrade --reinstall torch torchaudio

torch-check:
	$(UV) run --no-sync python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda'); print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"

test:
	MUZZLE_MODEL_BACKEND=fake $(UV) run --extra test pytest -q

dev: dev-real

dev-real: check-real-deps
	MUZZLE_MODEL_BACKEND=real $(UV) run --no-sync uvicorn muzzle.app:create_app --factory --host 127.0.0.1 --port $(PORT)

dev-fake:
	MUZZLE_MODEL_BACKEND=fake $(UV) run uvicorn muzzle.app:create_app --factory --reload --host 127.0.0.1 --port $(PORT)

example-tts:
	$(UV) run --no-sync python examples/stream_tts_pipewire.py --url "$(API_URL)" --text "$(TEXT)" --quality "$(QUALITY)" --player "$(PLAYER)" --latency-ms "$(LATENCY_MS)" $(TTS_KNOB_ARGS) $(PCM_ARGS)

example-tts-cpu:
	$(UV) run --no-sync python examples/stream_tts_cpu.py --text "$(TEXT)" --quality "$(CPU_QUALITY)" --player "$(PLAYER)" --latency-ms "$(LATENCY_MS)" $(TTS_KNOB_ARGS) $(PCM_ARGS)

example-stt:
	$(UV) run --no-sync python examples/stream_stt_pipewire.py --url "$(API_URL)" $(STT_ARGS)

example-pcm:
	$(UV) run --no-sync python examples/stream_tts_pipewire.py --url "$(API_URL)" --text "$(TEXT)" --quality "$(QUALITY)" --player none --save-pcm "$(if $(PCM_OUT),$(PCM_OUT),/tmp/muzzle-example.pcm)"

docker-build:
	docker build --target test-runtime --build-arg INSTALL_TEST=1 -t $(DOCKER_FAKE_IMAGE) .

docker-build-real: docker-build-cpu

docker-build-cpu:
	docker build --build-arg INSTALL_REAL=1 --build-arg INSTALL_TEST=0 --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu --build-arg TORCH_VERSION="$(DOCKER_TORCH_VERSION)" --build-arg TORCHAUDIO_VERSION="$(DOCKER_TORCHAUDIO_VERSION)" -t $(DOCKER_CPU_IMAGE) -t $(DOCKER_LATEST_IMAGE) .

docker-build-cu128:
	docker build -f Dockerfile.cu128 --build-arg CUDA_BASE_IMAGE="$(DOCKER_CU128_BASE_IMAGE)" --build-arg TORCH_VERSION="$(DOCKER_TORCH_VERSION)" --build-arg TORCHAUDIO_VERSION="$(DOCKER_TORCHAUDIO_VERSION)" -t $(DOCKER_CU128_IMAGE) .

docker-build-cu130:
	docker build -f Dockerfile.cu130 --build-arg CUDA_BASE_IMAGE="$(DOCKER_CU130_BASE_IMAGE)" --build-arg TORCH_VERSION="$(DOCKER_TORCH_VERSION)" --build-arg TORCHAUDIO_VERSION="$(DOCKER_TORCHAUDIO_VERSION)" -t $(DOCKER_CU130_IMAGE) .

docker-test: docker-build
	docker run --rm -e MUZZLE_MODEL_BACKEND=fake $(DOCKER_FAKE_IMAGE) pytest -q

docker-run-fake: docker-build
	docker run --rm -it -p $(PORT):8000 -e MUZZLE_MODEL_BACKEND=fake $(DOCKER_FAKE_IMAGE)

docker-run: docker-run-cpu

docker-run-real: docker-run-cpu

docker-run-cpu: docker-build-cpu
	docker run --rm -it -p $(PORT):8000 -e MUZZLE_MODEL_BACKEND=real -e MUZZLE_TTS_DEVICE=cpu -e MUZZLE_WHISPER_DEVICE=cpu $(DOCKER_CPU_IMAGE)

docker-run-cu128: docker-build-cu128
	docker run --rm -it --gpus all -p $(PORT):8000 $(DOCKER_CU128_IMAGE)

docker-run-cu130: docker-build-cu130
	docker run --rm -it --gpus all -p $(PORT):8000 $(DOCKER_CU130_IMAGE)
