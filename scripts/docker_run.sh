#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${DOCKER_IMAGE:-diffusion-trajectory-forecaster}"
WORKDIR_IN_CONTAINER="/app"
HOST_USERNAME="${USER:-$(id -un 2>/dev/null || echo user)}"

mkdir -p .docker-cache/uv .docker-cache/home

docker run --rm -it \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${PWD}:${WORKDIR_IN_CONTAINER}" \
  -v "${PWD}/.docker-cache/uv:/tmp/uv-cache" \
  -v "${PWD}/.docker-cache/home:/tmp/home" \
  -w "${WORKDIR_IN_CONTAINER}" \
  -e HOME=/tmp/home \
  -e USER="${HOST_USERNAME}" \
  -e LOGNAME="${HOST_USERNAME}" \
  -e UV_PROJECT_ENVIRONMENT=/opt/venv \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e HYDRA_FULL_ERROR=1 \
  -e MPLBACKEND=Agg \
  -e WANDB_MODE="${WANDB_MODE:-offline}" \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}" \
  -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}" \
  "${IMAGE_NAME}" \
  "$@"
