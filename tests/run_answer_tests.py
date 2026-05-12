from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES_DIR = Path(__file__).resolve().parent / "cases"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "results" / "answer_test_report.md"
DEFAULT_JSON_REPORT_PATH = Path(__file__).resolve().parent / "results" / "answer_test_report.json"
TOOL_ERROR_STATUSES = {"unavailable", "timeout", "http_error", "bad_json", "blocked"}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run black-box answer tests for VitalAI chatbot.")
    parser.add_argument("--category", default="all", help="Category file stem in tests/cases, or 'all'.")
    parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH), help="Markdown report output path.")
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_REPORT_PATH), help="JSON report output path.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed case.")
    return parser.parse_args()


def load_case_documents(category: str) -> list[dict[str, Any]]:
    if category == "all":
        paths = sorted(CASES_DIR.glob("*.json"))
    else:
        path = CASES_DIR / f"{category}.json"
        if not path.exists():
            raise FileNotFoundError(f"Category file not found: {path}")
        paths = [path]

    documents: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            documents.append(json.load(handle))
    return documents


async def run_case(answerer: Any, category: str, case: dict[str, Any]) -> dict[str, Any]:
    response = await answerer.answer(
        query=case["query"],
        top_k=int(case.get("top_k", 5)),
        disease_name=case.get("disease_name"),
        section_type=case.get("section_type"),
        source_type=case.get("source_type"),
        biomarker=case.get("biomarker"),
        include_debug=True,
    )
    evaluation = evaluate_case(case, response)
    return {
        "category": category,
        "id": case["id"],
        "title": case.get("title", case["id"]),
        "query": case["query"],
        "response": response,
        "evaluation": evaluation,
    }


def evaluate_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected", {})
    answer = str(response.get("answer") or "")
    answer_norm = normalize_text(answer)
    route = response.get("route")
    sources = response.get("sources") or []
    debug = response.get("debug") or {}
    router_plan = debug.get("router_plan") or {}
    medical_tool_result = debug.get("medical_tool_result") or {}
    extracted_tool_payload = debug.get("extracted_tool_payload") or {}
    formula_results = medical_tool_result.get("formula_results") or []
    threshold_matches = medical_tool_result.get("threshold_matches") or []
    classifications = medical_tool_result.get("classifications") or []

    checks: list[CheckResult] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(CheckResult(name=name, passed=passed, detail=detail))

    min_answer_chars = expected.get("min_answer_chars")
    if min_answer_chars is not None:
        add(
            "min_answer_chars",
            len(answer.strip()) >= int(min_answer_chars),
            f"actual={len(answer.strip())}, expected>={int(min_answer_chars)}",
        )

    require_route = expected.get("require_route")
    if require_route:
        add("require_route", route == require_route, f"actual={route}, expected={require_route}")

    require_sources_min = expected.get("require_sources_min")
    if require_sources_min is not None:
        add(
            "require_sources_min",
            len(sources) >= int(require_sources_min),
            f"actual={len(sources)}, expected>={int(require_sources_min)}",
        )

    if expected.get("require_tool_called"):
        tool_called = bool(router_plan.get("needs_medical_tool")) or bool(medical_tool_result)
        add("require_tool_called", tool_called, f"needs_medical_tool={router_plan.get('needs_medical_tool')}")

    if expected.get("require_tool_success"):
        status = medical_tool_result.get("tool_status")
        ok = bool(medical_tool_result) and status not in TOOL_ERROR_STATUSES
        add("require_tool_success", ok, f"tool_status={status}")

    if expected.get("require_no_null_tool_parameters"):
        params = (router_plan.get("tool_call") or {}).get("parameters") or {}
        ok = not contains_null(params)
        add("require_no_null_tool_parameters", ok, f"parameters={params}")

    require_extracted_measurements_min = expected.get("require_extracted_measurements_min")
    if require_extracted_measurements_min is not None:
        measurements = extracted_tool_payload.get("measurements") or {}
        add(
            "require_extracted_measurements_min",
            len(measurements) >= int(require_extracted_measurements_min),
            f"actual={len(measurements)}, expected>={int(require_extracted_measurements_min)}",
        )

    require_formula_result_min = expected.get("require_formula_result_min")
    if require_formula_result_min is not None:
        add(
            "require_formula_result_min",
            len(formula_results) >= int(require_formula_result_min),
            f"actual={len(formula_results)}, expected>={int(require_formula_result_min)}",
        )

    require_formula_ids_any = expected.get("require_formula_ids_any") or []
    if require_formula_ids_any:
        formula_ids = {str(item.get("formula_id") or "") for item in formula_results if isinstance(item, dict)}
        ok = any(item in formula_ids for item in require_formula_ids_any)
        add("require_formula_ids_any", ok, f"actual={sorted(formula_ids)}, expected_any={require_formula_ids_any}")

    require_threshold_match_min = expected.get("require_threshold_match_min")
    if require_threshold_match_min is not None:
        add(
            "require_threshold_match_min",
            len(threshold_matches) >= int(require_threshold_match_min),
            f"actual={len(threshold_matches)}, expected>={int(require_threshold_match_min)}",
        )

    require_classification_min = expected.get("require_classification_min")
    if require_classification_min is not None:
        add(
            "require_classification_min",
            len(classifications) >= int(require_classification_min),
            f"actual={len(classifications)}, expected>={int(require_classification_min)}",
        )

    must_include_any = [normalize_text(item) for item in expected.get("must_include_any", [])]
    if must_include_any:
        ok = any(token in answer_norm for token in must_include_any)
        add("must_include_any", ok, f"tokens={must_include_any}")

    must_include_all = [normalize_text(item) for item in expected.get("must_include_all", [])]
    if must_include_all:
        ok = all(token in answer_norm for token in must_include_all)
        add("must_include_all", ok, f"tokens={must_include_all}")

    must_not_include_any = [normalize_text(item) for item in expected.get("must_not_include_any", [])]
    if must_not_include_any:
        ok = all(token not in answer_norm for token in must_not_include_any)
        add("must_not_include_any", ok, f"tokens={must_not_include_any}")

    passed = all(check.passed for check in checks)
    return {
        "passed": passed,
        "checks": [check.__dict__ for check in checks],
    }


def normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def contains_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(contains_null(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_null(item) for item in value)
    return False


def build_markdown_report(results: list[dict[str, Any]], started_at: str) -> str:
    passed = sum(1 for item in results if item["evaluation"]["passed"])
    failed = len(results) - passed
    lines = [
        "# VitalAI Answer Test Report",
        "",
        f"- Generated at: `{started_at}`",
        f"- Total cases: `{len(results)}`",
        f"- Passed: `{passed}`",
        f"- Failed: `{failed}`",
        "",
    ]

    for item in results:
        status = "PASS" if item["evaluation"]["passed"] else "FAIL"
        lines.extend(
            [
                f"## [{status}] {item['id']} - {item['title']}",
                "",
                f"- Category: `{item['category']}`",
                f"- Query: `{item['query']}`",
                f"- Route: `{item['response'].get('route')}`",
                f"- Sources: `{len(item['response'].get('sources') or [])}`",
                "- Checks:",
            ]
        )
        for check in item["evaluation"]["checks"]:
            mark = "OK" if check["passed"] else "FAIL"
            lines.append(f"  - `{mark}` {check['name']}: {check['detail']}")
        lines.extend(
            [
                "- Answer:",
                "```text",
                (item["response"].get("answer") or "").strip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


async def main() -> int:
    args = parse_args()
    from src.LLM.qa.answering import build_answerer_from_env

    documents = load_case_documents(args.category)
    answerer = build_answerer_from_env()
    started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    results: list[dict[str, Any]] = []
    for document in documents:
        category = document.get("category", "unknown")
        for case in document.get("cases", []):
            result = await run_case(answerer, category, case)
            results.append(result)
            if args.fail_fast and not result["evaluation"]["passed"]:
                break
        if args.fail_fast and results and not results[-1]["evaluation"]["passed"]:
            break

    markdown_report = build_markdown_report(results, started_at)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_report, encoding="utf-8")

    json_output_path = Path(args.json_output)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(
        json.dumps(
            {
                "generated_at": started_at,
                "total_cases": len(results),
                "passed_cases": sum(1 for item in results if item["evaluation"]["passed"]),
                "failed_cases": sum(1 for item in results if not item["evaluation"]["passed"]),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Markdown report written to: {output_path}")
    print(f"JSON report written to: {json_output_path}")
    print(f"Passed {sum(1 for item in results if item['evaluation']['passed'])}/{len(results)} cases.")
    return 0 if all(item["evaluation"]["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
