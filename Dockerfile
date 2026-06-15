# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    MUZZLE_HOST=0.0.0.0 \
    MUZZLE_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

ARG UV_SYNC_EXTRAS="--extra test"
ARG REAL_BACKEND=0
ARG UV_TORCH_BACKEND=auto
ENV UV_TORCH_BACKEND=${UV_TORCH_BACKEND}

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project ${UV_SYNC_EXTRAS}

COPY README.md ./
COPY src ./src
COPY tests ./tests

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen ${UV_SYNC_EXTRAS}

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$REAL_BACKEND" = "1" ]; then \
        uv pip install --torch-backend "$UV_TORCH_BACKEND" --upgrade --reinstall torch torchaudio; \
    fi

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "muzzle.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
