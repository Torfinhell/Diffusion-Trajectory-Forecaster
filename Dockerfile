# syntax=docker/dockerfile:1.7

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=300 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/tmp/uv-cache \
    HOME=/tmp/home

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

RUN mkdir -p /tmp/home /tmp/uv-cache /opt/venv && chmod -R 777 /tmp/home /tmp/uv-cache /opt/venv

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/tmp/uv-cache uv sync --frozen --no-install-project

COPY . .
RUN --mount=type=cache,target=/tmp/uv-cache uv sync --frozen

ENV PATH="/opt/venv/bin:$PATH"

CMD ["bash"]
