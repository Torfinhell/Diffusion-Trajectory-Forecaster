# syntax=docker/dockerfile:1.7

# Runtime-only image: provides Python, system libs, uv, and GCloud CLI.
# The repo is git-pulled and deps are `uv sync`'d at container start on the
# target machine -- nothing app-specific is baked into the image, so it never
# goes stale and never bakes in secrets/local state.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=300 \
    # --- GPU memory behavior: make remote match intended local behavior ---
    # JAX otherwise preallocates ~75-90% of VRAM on first use, starving the
    # TF/grain data pipeline -> XLA retries 20GB->15GB->...->5GB before it fits.
    # Disable preallocation and cap JAX's share so the data pipeline has room.
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
    TF_FORCE_GPU_ALLOW_GROWTH=true

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Google Cloud CLI (per setup.ipynb instructions)
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
 && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

CMD ["bash"]
