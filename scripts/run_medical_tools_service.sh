#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

export MEDICAL_TOOLS_DATA_DIR="${MEDICAL_TOOLS_DATA_DIR:-data/processed_data}"
export MEDICAL_TOOLS_HOST="${MEDICAL_TOOLS_HOST:-0.0.0.0}"
export MEDICAL_TOOLS_PORT="${MEDICAL_TOOLS_PORT:-8010}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m uvicorn services.medical_tools.app:app \
  --host "${MEDICAL_TOOLS_HOST}" \
  --port "${MEDICAL_TOOLS_PORT}"
