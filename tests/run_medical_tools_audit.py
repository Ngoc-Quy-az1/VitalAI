from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.medical_tools.service import MedicalToolsService


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_REPORT_PATH = RESULTS_DIR / "medical_tools_audit_report.md"
DEFAULT_JSON_PATH = RESULTS_DIR / "medical_tools_audit_report.json"


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    title: str
    query: str
    formula_ids: list[str]
    disease_name: str | None
    checks: tuple[Callable[[dict[str, Any]], tuple[bool, str]], ...]


def formula_status(formula_id: str, expected_status: str) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    def check(result: dict[str, Any]) -> tuple[bool, str]:
        formula = _formula_by_id(result, formula_id)
        actual = formula.get("status") if formula else None
        return actual == expected_status, f"formula={formula_id}, actual={actual}, expected={expected_status}"

    return check


def formula_value(formula_id: str, expected: float, tolerance: float = 1e-4) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    def check(result: dict[str, Any]) -> tuple[bool, str]:
        formula = _formula_by_id(result, formula_id)
        actual = formula.get("value") if formula else None
        passed = actual is not None and abs(float(actual) - expected) <= tolerance
        return passed, f"formula={formula_id}, actual={actual}, expected={expected}±{tolerance}"

    return check


def has_classification(label: str) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    def check(result: dict[str, Any]) -> tuple[bool, str]:
        labels = _classification_labels(result)
        return label in labels, f"actual={labels}, expected_contains={label}"

    return check


def lacks_classification(label: str) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    def check(result: dict[str, Any]) -> tuple[bool, str]:
        labels = _classification_labels(result)
        return label not in labels, f"actual={labels}, expected_missing={label}"

    return check


def has_threshold_match(biomarker: str, label: str | None = None) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    def check(result: dict[str, Any]) -> tuple[bool, str]:
        matches = [
            item
            for item in result.get("threshold_matches", [])
            if item.get("biomarker") == biomarker
            and (label is None or (item.get("threshold") or {}).get("label") == label)
        ]
        return bool(matches), f"biomarker={biomarker}, label={label}, matched_count={len(matches)}"

    return check


def detected_measurement(name: str) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    def check(result: dict[str, Any]) -> tuple[bool, str]:
        names = [item.get("name") for item in result.get("detected_measurements", [])]
        return name in names, f"actual={names}, expected_contains={name}"

    return check


def no_detected_measurement(name: str) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    def check(result: dict[str, Any]) -> tuple[bool, str]:
        names = [item.get("name") for item in result.get("detected_measurements", [])]
        return name not in names, f"actual={names}, expected_missing={name}"

    return check


def _formula_by_id(result: dict[str, Any], formula_id: str) -> dict[str, Any] | None:
    for item in result.get("formula_results", []):
        if item.get("formula_id") == formula_id:
            return item
    return None


def _classification_labels(result: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for item in result.get("classifications", []):
        label = (item.get("threshold") or {}).get("label")
        if label:
            labels.append(str(label))
    return labels


def build_cases() -> list[AuditCase]:
    return [
        AuditCase(
            case_id="formula_ckd_epi_2021",
            title="CKD-EPI 2021 computes race-free eGFR",
            query="Nữ 60 tuổi, creatinine 1.4 mg/dL. Hãy ước tính eGFR.",
            formula_ids=["ckd_epi_2021_creatinine"],
            disease_name=None,
            checks=(
                formula_status("ckd_epi_2021_creatinine", "computed"),
                formula_value("ckd_epi_2021_creatinine", 43.0698),
                has_classification("G3b"),
            ),
        ),
        AuditCase(
            case_id="formula_mdrd_default_race",
            title="MDRD computes with default race assumption",
            query="Nữ 60 tuổi, creatinine 1.4 mg/dL. Hãy ước tính eGFR.",
            formula_ids=["mdrd_gfr"],
            disease_name=None,
            checks=(
                formula_status("mdrd_gfr", "computed"),
                formula_value("mdrd_gfr", 40.7681),
                has_classification("G3b"),
            ),
        ),
        AuditCase(
            case_id="formula_cockcroft_gault",
            title="Cockcroft-Gault computes expected creatinine clearance",
            query="Nam 65 tuổi, nặng 70 kg, creatinine 1.6 mg/dL. Tính Cockcroft-Gault.",
            formula_ids=["cockcroft_gault"],
            disease_name=None,
            checks=(
                formula_status("cockcroft_gault", "computed"),
                formula_value("cockcroft_gault", 45.5729),
            ),
        ),
        AuditCase(
            case_id="formula_bsa",
            title="Body surface area computes expected result",
            query="Cân nặng 55 kg, chiều cao 160 cm. Tính BSA.",
            formula_ids=["body_surface_area"],
            disease_name=None,
            checks=(
                formula_status("body_surface_area", "computed"),
                formula_value("body_surface_area", 1.5635),
            ),
        ),
        AuditCase(
            case_id="formula_fena_vietnamese",
            title="FENa computes from Vietnamese aliases",
            query="Natri niệu 20 mmol/L, natri máu 140 mmol/L, creatinine niệu 100 mg/dL, creatinine máu 1 mg/dL. Tính FENa.",
            formula_ids=["fena_formula"],
            disease_name="acute_kidney_injury",
            checks=(
                formula_status("fena_formula", "computed"),
                formula_value("fena_formula", 0.1429),
                has_classification("prerenal_aki_suggestive"),
            ),
        ),
        AuditCase(
            case_id="formula_fena_english_words",
            title="FENa computes from natural English wording",
            query="Urine Na 20 mmol/L, plasma Na 140 mmol/L, urine creatinine 100 mg/dL, plasma creatinine 1 mg/dL. Tính FENa.",
            formula_ids=["fena_formula"],
            disease_name="acute_kidney_injury",
            checks=(
                formula_status("fena_formula", "computed"),
                formula_value("fena_formula", 0.1429),
                no_detected_measurement("sodium"),
            ),
        ),
        AuditCase(
            case_id="acr_a1",
            title="ACR below 30 maps to A1",
            query="ACR 29 mg/g",
            formula_ids=[],
            disease_name=None,
            checks=(has_classification("A1"),),
        ),
        AuditCase(
            case_id="acr_a2",
            title="ACR from 30 to 299 maps to A2",
            query="ACR 299 mg/g",
            formula_ids=[],
            disease_name=None,
            checks=(has_classification("A2"),),
        ),
        AuditCase(
            case_id="acr_300_boundary",
            title="ACR 300 mg/g maps to A3",
            query="ACR 300 mg/g",
            formula_ids=[],
            disease_name=None,
            checks=(has_classification("A3"),),
        ),
        AuditCase(
            case_id="acr_ckd_context",
            title="ACR classification survives CKD disease filter",
            query="ACR 350 mg/g",
            formula_ids=[],
            disease_name="benh_than_man",
            checks=(has_classification("A3"),),
        ),
        AuditCase(
            case_id="gfr_g2",
            title="GFR 75 maps to G2",
            query="GFR 75 ml/ph/1.73m2",
            formula_ids=[],
            disease_name="benh_than_man",
            checks=(has_classification("G2"),),
        ),
        AuditCase(
            case_id="gfr_g3a",
            title="GFR 55 maps to G3a",
            query="GFR 55 ml/ph/1.73m2",
            formula_ids=[],
            disease_name="benh_than_man",
            checks=(has_classification("G3a"),),
        ),
        AuditCase(
            case_id="gfr_g5",
            title="GFR 10 maps to G5",
            query="GFR 10 ml/ph/1.73m2",
            formula_ids=[],
            disease_name="benh_than_man",
            checks=(has_classification("G5"),),
        ),
        AuditCase(
            case_id="proteinuria_threshold",
            title="Nephrotic-range proteinuria threshold matches",
            query="Protein niệu 24h 4 g/24h",
            formula_ids=[],
            disease_name="hoi_chung_than_hu",
            checks=(has_threshold_match("protein_niệu_24h"),),
        ),
        AuditCase(
            case_id="albumin_unit_conversion",
            title="Albumin converts g/dL to g/L before threshold compare",
            query="Albumin máu 2.8 g/dL",
            formula_ids=[],
            disease_name="hoi_chung_than_hu",
            checks=(has_threshold_match("albumin_máu"),),
        ),
        AuditCase(
            case_id="blood_pressure_parser",
            title="Blood pressure parser extracts systolic and diastolic values",
            query="Huyết áp 128/78 mmHg",
            formula_ids=[],
            disease_name="benh_than_man",
            checks=(
                detected_measurement("systolic_bp"),
                detected_measurement("diastolic_bp"),
                has_threshold_match("systolic_bp", "blood_pressure_target"),
                has_threshold_match("diastolic_bp", "blood_pressure_target"),
            ),
        ),
        AuditCase(
            case_id="hyperkalemia_boundary",
            title="Potassium 6.5 reaches severe hyperkalemia threshold",
            query="Kali 6.5 mmol/L",
            formula_ids=[],
            disease_name="acute_kidney_injury",
            checks=(has_classification("severe_hyperkalemia"),),
        ),
        AuditCase(
            case_id="hemoglobin_male_who",
            title="Male Hb 12 g/dL is recognized as below WHO threshold",
            query="Nam, Hb 12 g/dL",
            formula_ids=[],
            disease_name="benh_than_man",
            checks=(has_classification("anemia_threshold"),),
        ),
        AuditCase(
            case_id="hemoglobin_female_boundary",
            title="Female Hb 12 g/dL does not cross female anemia threshold",
            query="Nữ, Hb 12 g/dL",
            formula_ids=[],
            disease_name="benh_than_man",
            checks=(lacks_classification("anemia_threshold_female"),),
        ),
    ]


def run_audit() -> list[dict[str, Any]]:
    service = MedicalToolsService()
    results: list[dict[str, Any]] = []
    for case in build_cases():
        output = service.evaluate(
            text=case.query,
            disease_name=case.disease_name,
            formula_ids=case.formula_ids,
        )
        checks = []
        for check in case.checks:
            passed, detail = check(output)
            checks.append({"name": check.__name__, "passed": passed, "detail": detail})
        results.append(
            {
                "id": case.case_id,
                "title": case.title,
                "query": case.query,
                "formula_ids": case.formula_ids,
                "disease_name": case.disease_name,
                "passed": all(item["passed"] for item in checks),
                "checks": checks,
                "output_summary": {
                    "detected_measurements": output.get("detected_measurements", []),
                    "derived_measurements": output.get("derived_measurements", []),
                    "classifications": _classification_labels(output),
                    "formula_results": output.get("formula_results", []),
                },
            }
        )
    return results


def write_reports(results: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "generated_at": generated_at,
        "total": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "failed": sum(1 for item in results if not item["passed"]),
        "results": results,
    }
    DEFAULT_JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DEFAULT_REPORT_PATH.write_text(build_markdown_report(payload), encoding="utf-8")


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# VitalAI Medical Tools Audit Report",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Total cases: `{payload['total']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Failed: `{payload['failed']}`",
        "",
    ]
    for item in payload["results"]:
        status = "PASS" if item["passed"] else "FAIL"
        lines.extend(
            [
                f"## [{status}] {item['id']} - {item['title']}",
                "",
                f"- Query: `{item['query']}`",
                f"- disease_name: `{item['disease_name']}`",
                f"- formula_ids: `{item['formula_ids']}`",
                "- Checks:",
            ]
        )
        for check in item["checks"]:
            mark = "OK" if check["passed"] else "FAIL"
            lines.append(f"  - `{mark}` {check['detail']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    results = run_audit()
    write_reports(results)
    failed = sum(1 for item in results if not item["passed"])
    print(f"medical_tools_audit: total={len(results)} passed={len(results) - failed} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
