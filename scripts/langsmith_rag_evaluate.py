from __future__ import annotations

"""Run VitalAI RAG batch evaluation on LangSmith.

This script can evaluate either:

- `tests/cases/*.json` workflow/tool cases.
- `data/evaluate_data/qa_dataset_50_questions_enriched.json` RAG QA cases.
"""

import argparse
import asyncio
from collections import defaultdict
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langsmith import Client

from src.LLM.observability import configure_langsmith_from_env
from tests.run_answer_tests import evaluate_case


CASES_DIR = ROOT / "tests" / "cases"
DEFAULT_QA_DATASET_PATH = ROOT / "data" / "evaluate_data" / "qa_dataset_50_questions_enriched.json"
PROCESSED_DATA_DIR = ROOT / "data" / "processed_data"
DEFAULT_SUMMARY_PATH = ROOT / "tests" / "results" / "langsmith_rag_eval_summary.json"
DEFAULT_PROGRESS_PATH = ROOT / "tests" / "results" / "langsmith_rag_eval_progress.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LangSmith batch evaluation for VitalAI RAG.")
    parser.add_argument("--dataset", default="VitalAI RAG Eval Dev", help="LangSmith dataset name.")
    parser.add_argument(
        "--dataset-source",
        choices=["cases", "qa", "all"],
        default="cases",
        help="Use tests/cases, enriched QA dataset, or both.",
    )
    parser.add_argument("--category", default="all", help="Case file stem under tests/cases, or 'all'.")
    parser.add_argument("--qa-dataset-path", default=str(DEFAULT_QA_DATASET_PATH), help="Enriched QA dataset JSON path.")
    parser.add_argument("--experiment-prefix", default="vitalai-rag", help="LangSmith experiment prefix.")
    parser.add_argument("--max-concurrency", type=int, default=1, help="LangSmith target concurrency.")
    parser.add_argument("--num-repetitions", type=int, default=1, help="Repeat each example N times.")
    parser.add_argument(
        "--target-delay-seconds",
        type=float,
        default=float(os.getenv("LANGSMITH_TARGET_DELAY_SECONDS", "1.0")),
        help="Minimum delay before each target/RAG call.",
    )
    parser.add_argument(
        "--target-max-attempts",
        type=int,
        default=int(os.getenv("LANGSMITH_TARGET_MAX_ATTEMPTS", "3")),
        help="Retry target/RAG calls when rate limited.",
    )
    parser.add_argument(
        "--target-backoff-seconds",
        type=float,
        default=float(os.getenv("LANGSMITH_TARGET_BACKOFF_SECONDS", "20.0")),
        help="Base backoff for target/RAG retries after rate limit errors.",
    )
    parser.add_argument("--skip-upload", action="store_true", help="Do not upload local cases to LangSmith dataset.")
    parser.add_argument("--upload-only", action="store_true", help="Upload examples then exit without running eval.")
    parser.add_argument(
        "--upload-policy",
        choices=["update-existing", "skip-existing", "fail"],
        default="update-existing",
        help="How to handle examples that already exist in the LangSmith dataset.",
    )
    parser.add_argument("--export-jsonl", help="Write LangSmith examples to a local JSONL file and continue.")
    parser.add_argument("--use-llm-judges", action="store_true", help="Enable Mistral LLM-as-judge evaluators.")
    parser.add_argument("--judge-model", default=os.getenv("LANGSMITH_EVAL_MODEL", "mistral-large-latest"))
    parser.add_argument(
        "--judge-delay-seconds",
        type=float,
        default=float(os.getenv("LANGSMITH_JUDGE_DELAY_SECONDS", "5.0")),
        help="Minimum delay before each Mistral judge call.",
    )
    parser.add_argument(
        "--judge-max-attempts",
        type=int,
        default=int(os.getenv("LANGSMITH_JUDGE_MAX_ATTEMPTS", "4")),
        help="Retry Mistral judge calls when rate limited.",
    )
    parser.add_argument(
        "--judge-backoff-seconds",
        type=float,
        default=float(os.getenv("LANGSMITH_JUDGE_BACKOFF_SECONDS", "30.0")),
        help="Base backoff for judge retries after rate limit errors.",
    )
    parser.add_argument(
        "--summary-json",
        default=os.getenv("LANGSMITH_SUMMARY_JSON", str(DEFAULT_SUMMARY_PATH)),
        help="Write final average metric scores to this JSON file.",
    )
    parser.add_argument(
        "--progress-json",
        default=os.getenv("LANGSMITH_PROGRESS_JSON", str(DEFAULT_PROGRESS_PATH)),
        help="Track baseline, latest run, and change notes in this JSON file.",
    )
    parser.add_argument(
        "--change-note",
        default=os.getenv("LANGSMITH_CHANGE_NOTE", ""),
        help="Optional note describing what changed before this eval run.",
    )
    return parser.parse_args()


def load_case_documents(category: str) -> list[dict[str, Any]]:
    if category == "all":
        paths = sorted(CASES_DIR.glob("*.json"))
    else:
        path = CASES_DIR / f"{category}.json"
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy case file: {path}")
        paths = [path]
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def load_documents_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    if args.dataset_source in {"cases", "all"}:
        documents.extend(load_case_documents(args.category))
    if args.dataset_source in {"qa", "all"}:
        qa_path = Path(args.qa_dataset_path)
        if not qa_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy QA dataset: {qa_path}. Chạy scripts/prepare_langsmith_eval_data.py trước."
            )
        documents.append(json.loads(qa_path.read_text(encoding="utf-8")))
    return documents


def iter_case_examples(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for document in documents:
        category = document.get("category", "unknown")
        for case in document.get("cases", []):
            inputs = {
                "query": case["query"],
                "top_k": int(case.get("top_k", 5)),
                "disease_name": case.get("disease_name"),
                "section_type": case.get("section_type"),
                "source_type": case.get("source_type"),
                "biomarker": case.get("biomarker"),
            }
            outputs = {
                "expected": case.get("expected", {}),
                "reference_answer": case.get("reference_answer"),
                "reference_context": case.get("reference_context"),
                "required_facts": case.get("required_facts", []),
                "relevant_document_ids": case.get("relevant_document_ids", []),
                "relevant_source_ids": case.get("relevant_source_ids", []),
                "source_evidence": case.get("source_evidence", []),
            }
            metadata = {
                "case_id": case["id"],
                "category": category,
                "title": case.get("title", case["id"]),
                "description": document.get("description"),
                "tags": case.get("eval_tags", []),
                "eval_notes": case.get("eval_notes"),
            }
            example_id = uuid.uuid5(uuid.NAMESPACE_URL, f"vitalai-rag-eval:{category}:{case['id']}")
            examples.append({"id": str(example_id), "inputs": inputs, "outputs": outputs, "metadata": metadata})
    return examples


def ensure_dataset(client: Client, dataset_name: str) -> Any:
    try:
        return client.read_dataset(dataset_name=dataset_name)
    except Exception:
        return client.create_dataset(
            dataset_name=dataset_name,
            description="VitalAI RAG eval cases converted from tests/cases/*.json",
            metadata={"owner": "VitalAI", "created_by": "scripts/langsmith_rag_evaluate.py"},
        )


def upload_examples(
    client: Client,
    dataset_name: str,
    examples: list[dict[str, Any]],
    upload_policy: str = "update-existing",
) -> dict[str, int]:
    dataset = ensure_dataset(client, dataset_name)
    if not examples:
        return {"created": 0, "updated": 0, "skipped": 0, "total": 0}

    if upload_policy == "fail":
        client.create_examples(dataset_id=dataset.id, examples=examples)
        return {"created": len(examples), "updated": 0, "skipped": 0, "total": len(examples)}

    existing_ids = {str(example.id) for example in client.list_examples(dataset_id=dataset.id)}
    new_examples = [example for example in examples if str(example["id"]) not in existing_ids]
    existing_examples = [example for example in examples if str(example["id"]) in existing_ids]

    if new_examples:
        client.create_examples(dataset_id=dataset.id, examples=new_examples)

    if upload_policy == "skip-existing":
        return {
            "created": len(new_examples),
            "updated": 0,
            "skipped": len(existing_examples),
            "total": len(examples),
        }

    for example in existing_examples:
        client.update_example(
            example["id"],
            dataset_id=dataset.id,
            inputs=example.get("inputs") or {},
            outputs=example.get("outputs") or {},
            metadata=example.get("metadata") or {},
        )

    return {
        "created": len(new_examples),
        "updated": len(existing_examples),
        "skipped": 0,
        "total": len(examples),
    }


def export_examples_jsonl(path: str | Path, examples: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(example, ensure_ascii=False) for example in examples) + "\n",
        encoding="utf-8",
    )


class ApiThrottle:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = max(0.0, float(delay_seconds or 0.0))
        self._last_call_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.delay_seconds <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_seconds = self.delay_seconds - (now - self._last_call_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_call_at = time.monotonic()


def retry_rate_limited_call(
    call: Callable[[], Any],
    *,
    label: str,
    max_attempts: int,
    backoff_seconds: float,
) -> Any:
    attempts = max(1, int(max_attempts or 1))
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt >= attempts:
                raise
            sleep_seconds = max(0.0, float(backoff_seconds or 0.0)) * attempt
            print(
                f"{label} bị rate limit/capacity 429, chờ {sleep_seconds:.1f}s "
                f"rồi thử lại ({attempt}/{attempts})."
            )
            time.sleep(sleep_seconds)


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    return (
        status_code == 429
        or " 429" in text
        or "status 429" in text
        or "rate limit" in text
        or "rate_limited" in text
        or "capacity exceeded" in text
        or "service_tier_capacity_exceeded" in text
    )


def build_target(
    *,
    delay_seconds: float = 0.0,
    max_attempts: int = 3,
    backoff_seconds: float = 20.0,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    from src.LLM.qa.answering import build_answerer_from_env

    answerer = build_answerer_from_env()
    throttle = ApiThrottle(delay_seconds)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            response = retry_rate_limited_call(
                lambda: _run_answerer_once(answerer, inputs, throttle),
                label="Target/RAG call",
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise
            return _target_error_output(inputs, exc)

        debug = response.get("debug") or {}
        rag_documents = _documents_from_debug_results(debug.get("results") or [])
        tool_documents = _documents_from_medical_tool_result(debug.get("medical_tool_result") or {})
        documents = _dedupe_documents([*rag_documents, *tool_documents])
        answer = str(response.get("answer") or "")
        contexts = [item["content"] for item in documents if item.get("content")]
        rag_contexts = [item["content"] for item in rag_documents if item.get("content")]
        tool_contexts = [item["content"] for item in tool_documents if item.get("content")]
        return {
            "query": response.get("query"),
            "answer": answer,
            "output": answer,
            "content": answer,
            "context": "\n\n".join(contexts),
            "route": response.get("route"),
            "sources": response.get("sources", []),
            "documents": documents,
            "rag_documents": rag_documents,
            "tool_documents": tool_documents,
            "contexts": contexts,
            "rag_contexts": rag_contexts,
            "tool_contexts": tool_contexts,
            "router_plan": debug.get("router_plan"),
            "router_error": debug.get("router_error"),
            "medical_tool_result": debug.get("medical_tool_result"),
            "extracted_tool_payload": debug.get("extracted_tool_payload"),
            "filters": debug.get("filters"),
            "query_understanding": debug.get("query_understanding"),
        }

    return target


def _run_answerer_once(answerer: Any, inputs: dict[str, Any], throttle: ApiThrottle) -> dict[str, Any]:
    throttle.wait()
    return asyncio.run(
        answerer.answer(
            query=inputs["query"],
            top_k=int(inputs.get("top_k") or 5),
            disease_name=inputs.get("disease_name"),
            section_type=inputs.get("section_type"),
            source_type=inputs.get("source_type"),
            biomarker=inputs.get("biomarker"),
            include_debug=True,
        )
    )


def _target_error_output(inputs: dict[str, Any], exc: Exception) -> dict[str, Any]:
    error = f"{type(exc).__name__}: {str(exc)[:500]}"
    return {
        "query": inputs.get("query"),
        "answer": "",
        "output": "",
        "content": "",
        "context": "",
        "route": None,
        "sources": [],
        "documents": [],
        "rag_documents": [],
        "tool_documents": [],
        "contexts": [],
        "rag_contexts": [],
        "tool_contexts": [],
        "router_plan": None,
        "router_error": error,
        "medical_tool_result": None,
        "extracted_tool_payload": None,
        "filters": None,
        "query_understanding": None,
        "target_error": error,
    }


def _documents_from_debug_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        documents.append(
            {
                "document_id": item.get("document_id"),
                "source_id": item.get("source_id"),
                "source_type": item.get("source_type"),
                "section_type": item.get("section_type"),
                "disease_name": item.get("disease_name"),
                "biomarker": item.get("biomarker"),
                "content": item.get("content") or item.get("preview") or "",
                "preview": item.get("preview") or "",
                "similarity": item.get("similarity"),
                "keyword_score": item.get("keyword_score"),
            }
        )
    return documents


def _documents_from_medical_tool_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    threshold_catalog = _load_threshold_catalog()
    formula_catalog = _load_formula_catalog()

    for item in result.get("formula_results") or []:
        if not isinstance(item, dict):
            continue
        formula_id = item.get("formula_id")
        if not formula_id:
            continue
        catalog_item = formula_catalog.get(str(formula_id), {})
        documents.append(
            {
                "document_id": f"formula::{formula_id}",
                "source_id": str(formula_id),
                "source_type": "tool_formula",
                "section_type": catalog_item.get("section_type"),
                "disease_name": catalog_item.get("disease_name"),
                "biomarker": None,
                "content": _formula_tool_content(item, catalog_item),
                "preview": _formula_tool_content(item, catalog_item)[:300],
                "similarity": None,
                "keyword_score": None,
            }
        )

    for group_name in ("threshold_matches", "classifications"):
        for item in result.get(group_name) or []:
            if not isinstance(item, dict):
                continue
            threshold = item.get("threshold") or {}
            threshold_id = threshold.get("threshold_id")
            if not threshold_id:
                continue
            catalog_item = threshold_catalog.get(str(threshold_id), {})
            source = item.get("source") or {}
            content = _threshold_tool_content(item, catalog_item)
            documents.append(
                {
                    "document_id": f"threshold::{threshold_id}",
                    "source_id": str(threshold_id),
                    "source_type": "tool_threshold",
                    "section_type": threshold.get("section_type") or catalog_item.get("section_type"),
                    "disease_name": threshold.get("disease_name") or catalog_item.get("disease_name"),
                    "biomarker": item.get("biomarker") or catalog_item.get("biomarker"),
                    "content": content,
                    "preview": content[:300],
                    "similarity": None,
                    "keyword_score": None,
                    "source_file": source.get("source_file") or catalog_item.get("source_file"),
                }
            )
    return _dedupe_documents(documents)


def _load_threshold_catalog() -> dict[str, dict[str, Any]]:
    if hasattr(_load_threshold_catalog, "_cache"):
        return getattr(_load_threshold_catalog, "_cache")
    catalog: dict[str, dict[str, Any]] = {}
    for path in [PROCESSED_DATA_DIR / "thresholds.jsonl", PROCESSED_DATA_DIR / "thresholds_extra.jsonl"]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            catalog[str(item["threshold_id"])] = item
    setattr(_load_threshold_catalog, "_cache", catalog)
    return catalog


def _load_formula_catalog() -> dict[str, dict[str, Any]]:
    if hasattr(_load_formula_catalog, "_cache"):
        return getattr(_load_formula_catalog, "_cache")
    path = PROCESSED_DATA_DIR / "formulas.json"
    formulas = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    catalog = {str(item["formula_id"]): item for item in formulas}
    setattr(_load_formula_catalog, "_cache", catalog)
    return catalog


def _formula_tool_content(result: dict[str, Any], catalog_item: dict[str, Any]) -> str:
    parts = [
        f"Công thức: {result.get('formula_name') or catalog_item.get('formula_name') or result.get('formula_id')}",
        f"formula_id: {result.get('formula_id')}",
        f"status: {result.get('status')}",
    ]
    if result.get("value") is not None:
        parts.append(f"kết quả: {result.get('value')} {result.get('unit') or ''}".strip())
    if result.get("assumptions"):
        parts.append("giả định: " + "; ".join(str(item) for item in result.get("assumptions") or []))
    if catalog_item.get("expression"):
        parts.append(f"biểu thức: {catalog_item.get('expression')}")
    if catalog_item.get("source_text"):
        parts.append(f"nguồn: {catalog_item.get('source_text')}")
    return ". ".join(str(part) for part in parts if part)


def _threshold_tool_content(result: dict[str, Any], catalog_item: dict[str, Any]) -> str:
    threshold = result.get("threshold") or {}
    source = result.get("source") or {}
    parts = [
        f"Chỉ số: {result.get('biomarker') or catalog_item.get('biomarker')}",
        f"input: {result.get('input_value')} {result.get('input_unit') or ''}".strip(),
        f"ngưỡng: {threshold.get('op')} {threshold.get('value')}",
    ]
    if threshold.get("value_min") is not None or threshold.get("value_max") is not None:
        parts.append(f"khoảng: {threshold.get('value_min')} - {threshold.get('value_max')}")
    if threshold.get("unit"):
        parts.append(f"đơn vị ngưỡng: {threshold.get('unit')}")
    if threshold.get("label"):
        parts.append(f"phân loại: {threshold.get('label')}")
    source_text = source.get("source_text") or catalog_item.get("source_text")
    if source_text:
        parts.append(f"nguồn: {source_text}")
    return ". ".join(str(part) for part in parts if part)


def _dedupe_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for doc in documents:
        key = (str(doc.get("document_id") or ""), str(doc.get("source_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def extract_answer(outputs: dict[str, Any]) -> str:
    """Normalize programmatic target output and LangSmith UI chat-model output."""

    raw_outputs = outputs
    outputs = as_dict(raw_outputs)
    if not outputs:
        return _text_from_value(raw_outputs)
    for key in ("answer", "output", "content", "text", "response", "result"):
        text = _text_from_value(outputs.get(key))
        if text:
            return text
    generations = outputs.get("generations")
    if isinstance(generations, list) and generations:
        text = _text_from_value(generations[0])
        if text:
            return text
    return ""


def extract_contexts(outputs: dict[str, Any], reference_outputs: dict[str, Any] | None = None) -> list[str]:
    contexts: list[str] = []
    outputs = as_dict(outputs)
    if outputs:
        for key in ("contexts", "rag_contexts", "tool_contexts"):
            value = outputs.get(key)
            if isinstance(value, list):
                contexts.extend(str(item) for item in value if str(item or "").strip())
            else:
                text = _text_from_value(value)
                if text:
                    contexts.append(text)
        for key in ("context", "rag_context", "tool_context"):
            text = _text_from_value(outputs.get(key))
            if text:
                contexts.append(text)
        for key in ("documents", "rag_documents", "tool_documents"):
            value = outputs.get(key)
            if isinstance(value, list):
                for doc in value:
                    if isinstance(doc, dict):
                        text = _text_from_value(doc.get("content") or doc.get("preview"))
                    else:
                        text = _text_from_value(doc)
                    if text:
                        contexts.append(text)

    if not contexts and isinstance(reference_outputs, dict):
        text = _text_from_value(reference_outputs.get("reference_context"))
        if text:
            contexts.append(text)
        for item in reference_outputs.get("source_evidence") or []:
            if isinstance(item, dict):
                text = _text_from_value(item.get("content") or item.get("content_preview") or item.get("preview"))
                if text:
                    contexts.append(text)

    return _dedupe_texts(contexts)


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "text", "answer", "output", "response", "result"):
            text = _text_from_value(value.get(key))
            if text:
                return text
        message = value.get("message")
        if message is not None:
            text = _text_from_value(message)
            if text:
                return text
        return ""
    if isinstance(value, list):
        parts = [_text_from_value(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    return str(value).strip()


def _dedupe_texts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = normalize(text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def contract_checks(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Reuse existing black-box expected checks as LangSmith metrics."""

    outputs = as_dict(outputs)
    reference_outputs = as_dict(reference_outputs)
    expected = reference_outputs.get("expected") or {}
    if not expected:
        return []

    case = {"query": inputs.get("query", ""), "expected": expected}
    response = {
        "answer": extract_answer(outputs),
        "route": outputs.get("route"),
        "sources": outputs.get("sources"),
        "debug": {
            "router_plan": outputs.get("router_plan"),
            "medical_tool_result": outputs.get("medical_tool_result"),
            "extracted_tool_payload": outputs.get("extracted_tool_payload"),
        },
    }
    evaluation = evaluate_case(case, response)
    metrics = [{"key": "contract_pass", "score": 1.0 if evaluation["passed"] else 0.0}]
    for check in evaluation.get("checks", []):
        metrics.append(
            {
                "key": f"check_{check['name']}",
                "score": 1.0 if check["passed"] else 0.0,
                "comment": check.get("detail", ""),
            }
        )
    return metrics


def gold_context_metrics(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic context precision/recall when gold IDs or facts exist."""

    outputs = as_dict(outputs)
    reference_outputs = as_dict(reference_outputs)
    gold_ids = {
        str(value)
        for value in [
            *(reference_outputs.get("relevant_document_ids") or []),
            *(reference_outputs.get("relevant_source_ids") or []),
        ]
        if value
    }

    metrics: list[dict[str, Any]] = []
    if gold_ids:
        metrics.extend(_gold_id_metrics("context", outputs.get("documents") or [], gold_ids))
        metrics.extend(_gold_id_metrics("rag_context", outputs.get("rag_documents") or [], gold_ids))

    required_facts = [str(item).strip() for item in reference_outputs.get("required_facts") or [] if str(item).strip()]
    if required_facts:
        context_text = normalize(" ".join(outputs.get("contexts") or []))
        rag_context_text = normalize(" ".join(outputs.get("rag_contexts") or []))
        covered = sum(1 for fact in required_facts if normalize(fact) in context_text)
        metrics.append({"key": "context_recall_required_facts_exact", "score": covered / len(required_facts)})
        rag_covered = sum(1 for fact in required_facts if normalize(fact) in rag_context_text)
        metrics.append({"key": "rag_context_recall_required_facts_exact", "score": rag_covered / len(required_facts)})

    if not metrics:
        metrics.append({"key": "gold_context_available", "score": 0.0, "comment": "Missing relevant ids or required facts."})
    return metrics


def _gold_id_metrics(prefix: str, documents: list[dict[str, Any]], gold_ids: set[str]) -> list[dict[str, Any]]:
    retrieved_ids = {
        str(value)
        for doc in documents
        for value in (doc.get("document_id"), doc.get("source_id"))
        if value
    }
    hits = retrieved_ids & gold_ids
    precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = len(hits) / len(gold_ids) if gold_ids else 0.0
    return [
        {"key": f"{prefix}_precision_gold_ids", "score": precision},
        {"key": f"{prefix}_recall_gold_ids", "score": recall},
    ]


def build_llm_judges(
    model: str,
    *,
    delay_seconds: float = 0.0,
    max_attempts: int = 4,
    backoff_seconds: float = 30.0,
) -> list[Callable[..., Any]]:
    api_key = os.getenv("MISTRAL_CLIENT_API_KEY")
    if not api_key:
        raise ValueError("Thiếu MISTRAL_CLIENT_API_KEY để chạy --use-llm-judges.")

    judge = ChatMistralAI(
        model=model,
        api_key=api_key,
        temperature=0,
        max_tokens=512,
        max_retries=6,
        max_concurrent_requests=1,
    )
    throttle = ApiThrottle(delay_seconds)

    def answer_relevance(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
        answer = extract_answer(outputs)
        if not answer:
            return {
                "key": "answer_relevance_missing_output",
                "score": 0.0,
                "comment": "Missing answer. Expected outputs.answer or outputs.output.content.",
            }
        try:
            return judge_score(
                judge,
                key="answer_relevance",
                instruction=(
                    "Score 0 to 1 whether the ANSWER directly and helpfully addresses the QUESTION. "
                    "Do not grade factual accuracy; only relevance and focus."
                ),
                payload={"QUESTION": inputs.get("query"), "ANSWER": answer},
                throttle=throttle,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
        except Exception as exc:
            return judge_error_metric("answer_relevance_judge_error", exc)

    def groundedness_and_hallucination(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        outputs = as_dict(outputs)
        reference_outputs = as_dict(reference_outputs)
        answer = extract_answer(outputs)
        contexts = extract_contexts(outputs, reference_outputs)
        context = "\n\n".join(contexts)
        tool_context = json.dumps(outputs.get("medical_tool_result") or {}, ensure_ascii=False)[:6000]
        if not answer:
            return [
                {
                    "key": "groundedness_missing_output",
                    "score": 0.0,
                    "comment": "Missing answer. Expected outputs.answer or outputs.output.content.",
                },
                {
                    "key": "hallucination_eval_skipped",
                    "score": 0.0,
                    "comment": "Skipped because answer output is empty.",
                },
            ]
        if not context and not tool_context.strip("{}"):
            return [
                {
                    "key": "groundedness_missing_context",
                    "score": 0.0,
                    "comment": "Missing context. Expected outputs.contexts/documents or reference_outputs.reference_context.",
                },
                {
                    "key": "hallucination_eval_skipped",
                    "score": 0.0,
                    "comment": "Skipped because grounding context is empty.",
                },
            ]
        try:
            grade = judge_score(
                judge,
                key="groundedness",
                instruction=(
                    "Score 0 to 1 whether the ANSWER is fully supported by CONTEXT and TOOL_RESULT. "
                    "Penalize unsupported diagnoses, medications, causes, tests, thresholds, or formula results."
                ),
                payload={
                    "QUESTION": inputs.get("query"),
                    "ANSWER": answer,
                    "CONTEXT": context[:10000],
                    "TOOL_RESULT": tool_context,
                },
                throttle=throttle,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
        except Exception as exc:
            return [judge_error_metric("groundedness_judge_error", exc)]
        grounded_score = float(grade.get("score") or 0.0)
        return [
            grade,
            {
                "key": "hallucination",
                "score": max(0.0, min(1.0, 1.0 - grounded_score)),
                "comment": grade.get("comment", ""),
            },
        ]

    def context_precision_llm(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
        outputs = as_dict(outputs)
        documents = outputs.get("documents") or []
        if not documents:
            contexts = extract_contexts(outputs)
            documents = [{"content": item} for item in contexts]
        if not documents:
            return {"key": "context_precision_llm", "score": 0.0, "comment": "No retrieved documents."}

        relevant = 0
        comments: list[str] = []
        for index, doc in enumerate(documents, start=1):
            try:
                grade = judge_score(
                    judge,
                    key="chunk_relevance",
                    instruction=(
                        "Return score 1 if CONTEXT_CHUNK contains information relevant to QUESTION, otherwise 0. "
                        "A chunk can be partially relevant and still score 1."
                    ),
                    payload={"QUESTION": inputs.get("query"), "CONTEXT_CHUNK": str(doc.get("content") or "")[:4000]},
                    throttle=throttle,
                    max_attempts=max_attempts,
                    backoff_seconds=backoff_seconds,
                )
            except Exception as exc:
                return judge_error_metric("context_precision_judge_error", exc)
            score = float(grade.get("score") or 0.0)
            if score >= 0.5:
                relevant += 1
            comments.append(f"doc{index}={score:.2f}")
        return {
            "key": "context_precision_llm",
            "score": relevant / len(documents),
            "comment": "; ".join(comments),
        }

    def context_recall_llm(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        outputs = as_dict(outputs)
        reference_outputs = as_dict(reference_outputs)
        reference_answer = str(reference_outputs.get("reference_answer") or "").strip()
        required_facts = reference_outputs.get("required_facts") or []
        if not reference_answer and not required_facts:
            return {"key": "context_recall_llm", "score": 0.0, "comment": "Missing reference_answer or required_facts."}
        contexts = extract_contexts(outputs, reference_outputs)
        if not contexts:
            return {"key": "context_recall_llm", "score": 0.0, "comment": "CONTEXT is empty."}
        try:
            return judge_score(
                judge,
                key="context_recall_llm",
                instruction=(
                    "Score 0 to 1 whether CONTEXT contains the information needed to produce the REFERENCE_ANSWER "
                    "and covers the REQUIRED_FACTS. Ignore answer style."
                ),
                payload={
                    "REFERENCE_ANSWER": reference_answer,
                    "REQUIRED_FACTS": json.dumps(required_facts, ensure_ascii=False),
                    "CONTEXT": "\n\n".join(contexts)[:10000],
                },
                throttle=throttle,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
        except Exception as exc:
            return judge_error_metric("context_recall_judge_error", exc)

    return [answer_relevance, groundedness_and_hallucination, context_precision_llm, context_recall_llm]


def judge_score(
    judge: ChatMistralAI,
    *,
    key: str,
    instruction: str,
    payload: dict[str, Any],
    throttle: ApiThrottle | None = None,
    max_attempts: int = 4,
    backoff_seconds: float = 30.0,
) -> dict[str, Any]:
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a strict evaluator for a Vietnamese medical RAG system. "
                "Return only JSON with fields: score (number from 0 to 1) and explanation (short string). "
                + instruction
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]

    def call_judge() -> Any:
        if throttle is not None:
            throttle.wait()
        return judge.invoke(prompt)

    response = retry_rate_limited_call(
        call_judge,
        label=f"Judge call ({key})",
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )
    parsed = parse_json_object(str(response.content))
    score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
    return {"key": key, "score": score, "comment": str(parsed.get("explanation") or "")}


def judge_error_metric(key: str, exc: Exception) -> dict[str, Any]:
    return {
        "key": key,
        "score": 0.0,
        "comment": f"{type(exc).__name__}: {str(exc)[:500]}",
    }


def parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"score": 0.0, "explanation": f"Judge returned non-JSON: {text[:200]}"}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"score": 0.0, "explanation": f"Judge JSON parse error: {exc}"}
    return parsed if isinstance(parsed, dict) else {"score": 0.0, "explanation": "Judge JSON was not object."}


def normalize(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def summarize_experiment_results(results: Any) -> dict[str, float]:
    metric_scores: dict[str, list[float]] = defaultdict(list)
    for row in results:
        evaluation_results = get_result_field(row, "evaluation_results")
        for item in iter_evaluation_items(evaluation_results):
            key = get_result_field(item, "key")
            score = numeric_score(get_result_field(item, "score"))
            if key and score is not None:
                metric_scores[str(key)].append(score)

    return {
        metric: round(sum(scores) / len(scores), 6)
        for metric, scores in sorted(metric_scores.items())
        if scores
    }


def summarize_experiment_feedback(client: Client, experiment_name: str) -> dict[str, float]:
    """Read uploaded feedback from LangSmith and compute weighted metric means."""

    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    run_ids: list[Any] = []

    for run in client.list_runs(
        project_name=experiment_name,
        is_root=True,
        run_type="chain",
        select=["id", "feedback_stats"],
    ):
        run_ids.append(run.id)
        feedback_stats = getattr(run, "feedback_stats", None) or {}
        for key, stats in feedback_stats.items():
            if not isinstance(stats, dict):
                continue
            avg = numeric_score(stats.get("avg"))
            n = int(stats.get("n") or 0)
            if avg is None or n <= 0:
                continue
            totals[str(key)] += avg * n
            counts[str(key)] += n

    if counts:
        return {
            metric: round(totals[metric] / counts[metric], 6)
            for metric in sorted(counts)
            if counts[metric] > 0
        }

    for batch in chunked(run_ids, 50):
        for feedback in client.list_feedback(run_ids=batch):
            key = getattr(feedback, "key", None)
            score = numeric_score(getattr(feedback, "score", None))
            if key and score is not None:
                totals[str(key)] += score
                counts[str(key)] += 1

    return {
        metric: round(totals[metric] / counts[metric], 6)
        for metric in sorted(counts)
        if counts[metric] > 0
    }


def chunked(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def iter_evaluation_items(evaluation_results: Any) -> list[Any]:
    if evaluation_results is None:
        return []
    if isinstance(evaluation_results, dict):
        raw_results = evaluation_results.get("results", [])
    else:
        raw_results = getattr(evaluation_results, "results", [])
    if raw_results is None:
        return []
    if isinstance(raw_results, list):
        return raw_results
    return [raw_results]


def get_result_field(item: Any, field: str) -> Any:
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)


def numeric_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str):
        try:
            score = float(value)
        except ValueError:
            return None
    else:
        return None
    return score if math.isfinite(score) else None


def write_summary_json(path: str | Path, summary: dict[str, float]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def update_progress_json(
    path: str | Path,
    *,
    summary: dict[str, float],
    experiment_name: str | None,
    started_at: str,
    dataset_name: str,
    dataset_source: str,
    judge_model: str | None,
    change_note: str,
) -> None:
    output_path = Path(path)
    if output_path.exists():
        data = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        data = {
            "baseline": {
                "recorded_at": started_at,
                "source": "first_eval_run",
                "metrics": summary,
            },
            "targets": {
                "answer_relevance": ">= 0.85",
                "groundedness": ">= 0.85",
                "hallucination": "<= 0.15",
                "context_precision_llm": ">= 0.55",
                "context_recall_llm": ">= 0.70",
            },
            "runs": [],
            "change_log": [],
        }

    data.setdefault("runs", [])
    data.setdefault("change_log", [])
    data.setdefault("baseline", {"recorded_at": started_at, "source": "first_eval_run", "metrics": summary})
    data["latest"] = {
        "recorded_at": started_at,
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "dataset_source": dataset_source,
        "judge_model": judge_model,
        "metrics": summary,
    }
    data["runs"].append(data["latest"])
    if change_note:
        data["change_log"].append(
            {
                "recorded_at": started_at,
                "experiment_name": experiment_name,
                "note": change_note,
                "metrics": summary,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    load_dotenv()
    configure_langsmith_from_env()
    args = parse_args()

    documents = load_documents_from_args(args)
    examples = iter_case_examples(documents)
    if args.export_jsonl:
        export_examples_jsonl(args.export_jsonl, examples)
        print(f"Exported {len(examples)} examples to JSONL: {args.export_jsonl}")
    if args.skip_upload and args.upload_only:
        return 0

    client = Client()
    if not args.skip_upload:
        upload_summary = upload_examples(client, args.dataset, examples, args.upload_policy)
        print(
            "Synced "
            f"{upload_summary['total']} examples to LangSmith dataset: {args.dataset} "
            f"(created={upload_summary['created']}, "
            f"updated={upload_summary['updated']}, "
            f"skipped={upload_summary['skipped']}, "
            f"policy={args.upload_policy})"
        )
    if args.upload_only:
        return 0

    evaluators: list[Callable[..., Any]] = [gold_context_metrics]
    if args.dataset_source in {"cases", "all"}:
        evaluators.insert(0, contract_checks)
    if args.use_llm_judges:
        evaluators.extend(
            build_llm_judges(
                args.judge_model,
                delay_seconds=args.judge_delay_seconds,
                max_attempts=args.judge_max_attempts,
                backoff_seconds=args.judge_backoff_seconds,
            )
        )

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = client.evaluate(
        build_target(
            delay_seconds=args.target_delay_seconds,
            max_attempts=args.target_max_attempts,
            backoff_seconds=args.target_backoff_seconds,
        ),
        data=args.dataset,
        evaluators=evaluators,
        experiment_prefix=args.experiment_prefix,
        max_concurrency=args.max_concurrency,
        num_repetitions=args.num_repetitions,
        metadata={
            "app": "VitalAI",
            "category": args.category,
            "dataset_source": args.dataset_source,
            "started_at": started_at,
            "llm_judges": args.use_llm_judges,
            "judge_model": args.judge_model if args.use_llm_judges else None,
        },
    )
    if hasattr(results, "wait"):
        results.wait()

    summary = summarize_experiment_results(results)
    experiment_name = getattr(results, "experiment_name", None)
    if not summary and experiment_name:
        summary = summarize_experiment_feedback(client, experiment_name)
    if not summary:
        raise RuntimeError(
            "Không thu được metric nào từ LangSmith evaluation; "
            "không ghi summary rỗng. Kiểm tra evaluator/feedback trong experiment."
        )

    write_summary_json(args.summary_json, summary)
    update_progress_json(
        args.progress_json,
        summary=summary,
        experiment_name=experiment_name,
        started_at=started_at,
        dataset_name=args.dataset,
        dataset_source=args.dataset_source,
        judge_model=args.judge_model if args.use_llm_judges else None,
        change_note=args.change_note,
    )
    print(f"LangSmith experiment started/completed: {results}")
    print(f"Wrote average metric summary to: {args.summary_json}")
    print(f"Updated eval progress log: {args.progress_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
