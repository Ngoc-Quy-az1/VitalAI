#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

export AI_SERVICE_HOST="${AI_SERVICE_HOST:-0.0.0.0}"
export AI_SERVICE_PORT="${AI_SERVICE_PORT:-8008}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m uvicorn src.api.app:app \
  --host "${AI_SERVICE_HOST}" \
  --port "${AI_SERVICE_PORT}"
