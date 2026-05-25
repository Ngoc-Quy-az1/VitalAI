#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-eval}"
DATASET_NAME="${LANGSMITH_DATASET_NAME:-vitalAI eval}"
DATASET_SOURCE="${LANGSMITH_DATASET_SOURCE:-qa}"
EXPERIMENT_PREFIX="${LANGSMITH_EXPERIMENT_PREFIX:-rag-eval-small-judge}"
MAX_CONCURRENCY="${LANGSMITH_MAX_CONCURRENCY:-1}"
UPLOAD_POLICY="${LANGSMITH_UPLOAD_POLICY:-update-existing}"
SUMMARY_JSON="${LANGSMITH_SUMMARY_JSON:-tests/results/langsmith_rag_eval_summary.json}"
PROGRESS_JSON="${LANGSMITH_PROGRESS_JSON:-tests/results/langsmith_rag_eval_progress.json}"
CHANGE_NOTE="${LANGSMITH_CHANGE_NOTE:-}"
TARGET_DELAY_SECONDS="${LANGSMITH_TARGET_DELAY_SECONDS:-1.0}"
TARGET_MAX_ATTEMPTS="${LANGSMITH_TARGET_MAX_ATTEMPTS:-3}"
TARGET_BACKOFF_SECONDS="${LANGSMITH_TARGET_BACKOFF_SECONDS:-20.0}"
JUDGE_DELAY_SECONDS="${LANGSMITH_JUDGE_DELAY_SECONDS:-5.0}"
JUDGE_MAX_ATTEMPTS="${LANGSMITH_JUDGE_MAX_ATTEMPTS:-4}"
JUDGE_BACKOFF_SECONDS="${LANGSMITH_JUDGE_BACKOFF_SECONDS:-30.0}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

require_runtime_deps() {
  "${PYTHON_BIN}" - <<'PY'
missing = []
for module in ("openai", "asyncpg", "langchain_mistralai", "langgraph", "langsmith"):
    try:
        __import__(module)
    except ModuleNotFoundError:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing Python modules: "
        + ", ".join(missing)
        + "\nRun: .venv/bin/python -m pip install -r requirements.txt"
    )
PY
}

case "${MODE}" in
  prepare)
    "${PYTHON_BIN}" scripts/prepare_langsmith_eval_data.py
    ;;
  upload)
    "${PYTHON_BIN}" scripts/langsmith_rag_evaluate.py \
      --dataset "${DATASET_NAME}" \
      --dataset-source "${DATASET_SOURCE}" \
      --upload-policy "${UPLOAD_POLICY}" \
      --upload-only
    ;;
  eval)
    require_runtime_deps
    "${PYTHON_BIN}" scripts/langsmith_rag_evaluate.py \
      --dataset "${DATASET_NAME}" \
      --dataset-source "${DATASET_SOURCE}" \
      --skip-upload \
      --experiment-prefix "${EXPERIMENT_PREFIX}" \
      --max-concurrency "${MAX_CONCURRENCY}" \
      --target-delay-seconds "${TARGET_DELAY_SECONDS}" \
      --target-max-attempts "${TARGET_MAX_ATTEMPTS}" \
      --target-backoff-seconds "${TARGET_BACKOFF_SECONDS}" \
      --judge-delay-seconds "${JUDGE_DELAY_SECONDS}" \
      --judge-max-attempts "${JUDGE_MAX_ATTEMPTS}" \
      --judge-backoff-seconds "${JUDGE_BACKOFF_SECONDS}" \
      --summary-json "${SUMMARY_JSON}" \
      --progress-json "${PROGRESS_JSON}" \
      --change-note "${CHANGE_NOTE}" \
      --use-llm-judges
    ;;
  all)
    require_runtime_deps
    "${PYTHON_BIN}" scripts/prepare_langsmith_eval_data.py
    "${PYTHON_BIN}" scripts/langsmith_rag_evaluate.py \
      --dataset "${DATASET_NAME}" \
      --dataset-source "${DATASET_SOURCE}" \
      --upload-policy "${UPLOAD_POLICY}" \
      --upload-only
    "${PYTHON_BIN}" scripts/langsmith_rag_evaluate.py \
      --dataset "${DATASET_NAME}" \
      --dataset-source "${DATASET_SOURCE}" \
      --skip-upload \
      --experiment-prefix "${EXPERIMENT_PREFIX}" \
      --max-concurrency "${MAX_CONCURRENCY}" \
      --target-delay-seconds "${TARGET_DELAY_SECONDS}" \
      --target-max-attempts "${TARGET_MAX_ATTEMPTS}" \
      --target-backoff-seconds "${TARGET_BACKOFF_SECONDS}" \
      --judge-delay-seconds "${JUDGE_DELAY_SECONDS}" \
      --judge-max-attempts "${JUDGE_MAX_ATTEMPTS}" \
      --judge-backoff-seconds "${JUDGE_BACKOFF_SECONDS}" \
      --summary-json "${SUMMARY_JSON}" \
      --progress-json "${PROGRESS_JSON}" \
      --change-note "${CHANGE_NOTE}" \
      --use-llm-judges
    ;;
  export)
    "${PYTHON_BIN}" scripts/langsmith_rag_evaluate.py \
      --dataset-source "${DATASET_SOURCE}" \
      --skip-upload \
      --upload-only \
      --export-jsonl "tests/results/langsmith_${DATASET_SOURCE}_examples.jsonl"
    ;;
  *)
    echo "Usage: $0 [prepare|upload|eval|all|export]" >&2
    echo "Env overrides: LANGSMITH_DATASET_NAME, LANGSMITH_DATASET_SOURCE, LANGSMITH_EXPERIMENT_PREFIX, LANGSMITH_MAX_CONCURRENCY, LANGSMITH_UPLOAD_POLICY, LANGSMITH_SUMMARY_JSON, LANGSMITH_PROGRESS_JSON, LANGSMITH_CHANGE_NOTE, LANGSMITH_TARGET_DELAY_SECONDS, LANGSMITH_TARGET_MAX_ATTEMPTS, LANGSMITH_TARGET_BACKOFF_SECONDS, LANGSMITH_JUDGE_DELAY_SECONDS, LANGSMITH_JUDGE_MAX_ATTEMPTS, LANGSMITH_JUDGE_BACKOFF_SECONDS" >&2
    exit 2
    ;;
esac
