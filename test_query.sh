#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/scripts/test_answer.sh" --query "Lupus ban đỏ là gì?" --top-k 5
# "${SCRIPT_DIR}/scripts/test_retrieval.sh" --query "Lupus ban đỏ là gì?" --top-k 5
# "${SCRIPT_DIR}/scripts/test_answer.sh" --query "Công thức Cockcroft-Gault là gì?" --top-k 5
# "${SCRIPT_DIR}/scripts/test_answer.sh" --query "Công thức Cockcroft-Gault là gì?" --source-type formula --top-k 3
