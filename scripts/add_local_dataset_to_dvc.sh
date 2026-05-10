#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <local-dataset-path>" >&2
  echo "Example: $0 data/small_base_v2_no_scenes" >&2
  exit 1
fi

DATASET_PATH="$1"
DVC_FILE="${DATASET_PATH%/}.dvc"

uv run dvc add "${DATASET_PATH}"
uv run dvc push "${DVC_FILE}"
git add "${DVC_FILE}" .gitignore

echo "Added ${DATASET_PATH} to DVC"
echo "Pushed ${DVC_FILE}"
echo "Staged ${DVC_FILE} and .gitignore"
