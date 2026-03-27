#!/usr/bin/env bash
set -euo pipefail

REMOTE_NAME="${1:-myremote}"
REMOTE_URL="${2:-}"

if [[ -z "${REMOTE_URL}" ]]; then
  echo "Usage: $0 <remote-name> <remote-url>" >&2
  echo "Example: $0 myremote ssh://user@host/path/to/dvc-store" >&2
  exit 1
fi

if [[ ! -d .dvc ]]; then
  dvc init
fi

dvc remote add -f "${REMOTE_NAME}" "${REMOTE_URL}"
dvc remote default "${REMOTE_NAME}"
git add .dvc/config

echo "Configured DVC remote '${REMOTE_NAME}' for ${REMOTE_URL}"
