# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm AS builder
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

ARG INSTALL_REAL=0
ARG INSTALL_TEST=1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG CHATTERBOX_TTS_REQUIREMENT="chatterbox-tts @ git+https://github.com/chrishayen/chatterbox.git@ce5a900dbdbc776eeb3d00c1c1607143c7604a5b"
ARG PERTH_REQUIREMENT="resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master"
ARG S3TOKENIZER_REQUIREMENT=s3tokenizer
ARG SILERO_VAD_REQUIREMENT="silero-vad>=6.0.0"
ARG CONFORMER_REQUIREMENT="conformer==0.3.2"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip setuptools wheel

COPY pyproject.toml README.md ./
COPY src ./src
COPY docker/requirements-real.txt ./docker/requirements-real.txt

RUN if [ "$INSTALL_TEST" = "1" ]; then \
        python -m pip install ".[test]"; \
    else \
        python -m pip install .; \
    fi \
    && if [ "$INSTALL_REAL" = "1" ]; then \
        python -m pip install --index-url "$PYTORCH_INDEX_URL" "torch==$TORCH_VERSION" "torchaudio==$TORCHAUDIO_VERSION"; \
        python -m pip install -r docker/requirements-real.txt; \
        python -m pip install --no-deps "$SILERO_VAD_REQUIREMENT" "$CONFORMER_REQUIREMENT"; \
        python -m pip install --no-deps "$S3TOKENIZER_REQUIREMENT"; \
        python -m pip install --no-deps "$PERTH_REQUIREMENT"; \
        python -m pip install --no-deps "$CHATTERBOX_TTS_REQUIREMENT"; \
    fi

RUN rm -rf \
        /opt/venv/lib/python*/site-packages/torch/include \
        /opt/venv/lib/python*/site-packages/torch/share/cmake \
        /opt/venv/lib/python*/site-packages/torch/test \
    && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} +

FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    MUZZLE_HOST=0.0.0.0 \
    MUZZLE_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY README.md ./
COPY src ./src

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "muzzle.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test-runtime
COPY tests ./tests

FROM runtime AS default
