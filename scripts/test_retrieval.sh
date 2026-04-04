#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

if [[ $# -eq 0 ]]; then
  echo " Lupus ban do la gi?" >&2
  run_repo_python scripts/test_retrieval.py --query "Lupus ban đỏ là gì?" --top-k 5
else
  run_repo_python scripts/test_retrieval.py "$@"
fi
