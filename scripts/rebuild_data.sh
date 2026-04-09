#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/process_data.sh" "$@"
"${SCRIPT_DIR}/prepare_embedding.sh"
