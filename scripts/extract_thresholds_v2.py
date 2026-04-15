from __future__ import annotations

"""Extract bổ sung threshold từ prose chunks.

Extractor gốc đang bắt tốt nhóm biomarker thận chính như GFR/ACR/PCR/protein niệu,
nhưng bỏ sót nhiều numeric rule viết trong prose: Hb/Hct, pH, bicarbonate, HbA1c,
huyết áp, phospho, FENa... Script này tạo file bổ sung `thresholds_extra.jsonl`
thay vì ghi đè `thresholds.jsonl` để audit dễ hơn.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract thresholds bổ sung từ chunks.jsonl.")
    parser.add_argument("--input", default="data/processed_data/chunks.jsonl")
    parser.add_argument("--base-thresholds", default="data/processed_data/thresholds.jsonl")
    parser.add_argument("--output", default="data/processed_data/thresholds_extra.jsonl")
    args = parser.parse_args()

    chunks = _load_jsonl(Path(args.input))
    base_thresholds = _load_jsonl(Path(args.base_thresholds))
    base_keys = {_dedupe_key(item) for item in base_thresholds}

    extracted: list[dict[str, Any]] = []
    for chunk in chunks:
        extracted.extend(_extract_from_chunk(chunk))

    deduped: list[dict[str, Any]] = []
    seen = set(base_keys)
    for item in extracted:
        key = _dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        item["threshold_id"] = _make_threshold_id(item, len(deduped) + 1)
        deduped.append(item)

    output = Path(args.output)
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in deduped), encoding="utf-8")
    print(json.dumps({"output": str(output), "extra_thresholds": len(deduped)}, ensure_ascii=False, indent=2))


def _extract_from_chunk(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    text = chunk["content"]
    metadata = chunk["metadata"]
    results: list[dict[str, Any]] = []

    results.extend(_simple_matches(chunk, r"Hb\s*<\s*(?P<value>\d+(?:[.,]\d+)?)\s*g/?l", "hemoglobin", "<", "g/L", "anemia_threshold"))
    results.extend(_simple_matches(chunk, r"(?:và|,)\s*<\s*(?P<value>\d+(?:[.,]\d+)?)\s*g/?l\s*ở\s*nữ", "hemoglobin", "<", "g/L", "anemia_threshold_female"))
    results.extend(_simple_matches(chunk, r"Hct\s*<\s*(?P<value>\d+(?:[.,]\d+)?)\s*%?", "hematocrit", "<", "%", "hematocrit_low"))
    results.extend(_simple_matches(chunk, r"pH\s*<\s*(?P<value>\d+(?:[.,]\d+)?)", "pH", "<", None, "metabolic_acidosis"))
    results.extend(_simple_matches(chunk, r"Bicarbonat\s*<\s*(?P<value>\d+(?:[.,]\d+)?)\s*mmol/?l", "bicarbonate", "<", "mmol/L", "severe_metabolic_acidosis"))
    results.extend(_simple_matches(chunk, r"HbA1C?\s*(?:dưới|duoi|<)\s*(?P<value>\d+(?:[.,]\d+)?)\s*%", "HbA1c", "<", "%", "glycemic_target"))
    results.extend(_simple_matches(chunk, r"LDL\s*cholesterol\s*<\s*(?P<value>\d+(?:[.,]\d+)?)\s*mg/?dl", "LDL_cholesterol", "<", "mg/dL", "lipid_target"))
    results.extend(_simple_matches(chunk, r"Na(?:\+|⁺)?\s*<\s*(?P<value>\d+(?:[.,]\d+)?)\s*mmol/?l", "sodium", "<", "mmol/L", "severe_hyponatremia"))
    results.extend(_simple_matches(chunk, r"kali\s+máu\s*[≥>=]+\s*(?P<value>\d+(?:[.,]\d+)?)\s*mmol/?l", "potassium", ">=", "mmol/L", "severe_hyperkalemia"))
    results.extend(_simple_matches(chunk, r"creatinin\s+huyết\s+thanh\s*>\s*(?P<value>\d+(?:[.,]\d+)?)\s*(?:μmol/l|µmol/l|umol/l)", "creatinine_umol_l", ">", "μmol/L", "creatinine_high"))
    results.extend(_simple_matches(chunk, r"gia\s+tăng\s+(?:nồng\s+độ\s+)?creatinin\s+huyết\s+thanh\s*>\s*(?P<value>\d+(?:[.,]\d+)?)\s*(?:μmol/l|µmol/l|umol/l)", "creatinine_change_umol_l", ">", "μmol/L", "creatinine_rise"))
    results.extend(_simple_matches(chunk, r"FENa\s*<\s*(?P<value>\d+(?:[.,]\d+)?)\s*%", "FENa", "<", "%", "prerenal_aki_suggestive"))
    results.extend(_simple_matches(chunk, r"FENa\s*>\s*(?P<value>\d+(?:[.,]\d+)?)\s*%", "FENa", ">", "%", "intrinsic_aki_suggestive"))

    for match in re.finditer(r"<\s*(?P<systolic>\d{2,3})\s*/\s*(?P<diastolic>\d{2,3})\s*mmHg", text, flags=re.IGNORECASE):
        source_text = _window(text, match.start(), match.end())
        results.append(_build_threshold(chunk, "systolic_bp", "<", _to_float(match.group("systolic")), "mmHg", "blood_pressure_target", source_text))
        results.append(_build_threshold(chunk, "diastolic_bp", "<", _to_float(match.group("diastolic")), "mmHg", "blood_pressure_target", source_text))

    phosphorus = re.search(
        r"(?P<min>\d+(?:[.,]\d+)?)\s*mg/?dl\s*\([^)]*\)\s*<\s*P\s*<\s*(?P<max>\d+(?:[.,]\d+)?)\s*mg/?dl",
        text,
        flags=re.IGNORECASE,
    )
    if phosphorus:
        item = _build_threshold(
            chunk,
            "phosphorus",
            "between",
            _to_float(phosphorus.group("min")),
            "mg/dL",
            "phosphorus_target",
            _window(text, phosphorus.start(), phosphorus.end()),
        )
        item["threshold_value_min"] = _to_float(phosphorus.group("min"))
        item["threshold_value_max"] = _to_float(phosphorus.group("max"))
        results.append(item)

    return [item for item in results if item.get("threshold_value") is not None and _looks_medical_threshold(item, metadata)]


def _simple_matches(
    chunk: dict[str, Any],
    pattern: str,
    biomarker: str,
    op: str,
    unit: str | None,
    label: str,
) -> Iterable[dict[str, Any]]:
    text = chunk["content"]
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        yield _build_threshold(chunk, biomarker, op, _to_float(match.group("value")), unit, label, _window(text, match.start(), match.end()))


def _build_threshold(
    chunk: dict[str, Any],
    biomarker: str,
    op: str,
    value: float | None,
    unit: str | None,
    label: str,
    source_text: str,
) -> dict[str, Any]:
    metadata = chunk["metadata"]
    return {
        "threshold_id": "",
        "biomarker": biomarker,
        "threshold_op": op,
        "threshold_value": value,
        "threshold_unit": unit,
        "label": label,
        "severity": None,
        "disease_name": metadata.get("disease_name"),
        "section_type": metadata.get("section_type"),
        "content_type": "threshold_value",
        "source_text": source_text,
        "source_file": metadata.get("source_file"),
        "page": metadata.get("page"),
        "language": metadata.get("language", "vi"),
        "extraction_method": "thresholds_v2_targeted",
    }


def _looks_medical_threshold(item: dict[str, Any], metadata: dict[str, Any]) -> bool:
    source_text = item.get("source_text") or ""
    if source_text.count("}") > 2:
        return False
    if item.get("biomarker") == "creatinine_umol_l" and "gia tăng" in source_text.lower():
        return False
    if item.get("biomarker") in {"hemoglobin", "hematocrit", "pH", "bicarbonate", "HbA1c", "LDL_cholesterol"}:
        return True
    if metadata.get("section_type") in {"diagnosis_criteria", "classification", "treatment", "clinical_features", "general"}:
        return True
    return False


def _dedupe_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item.get("biomarker"),
        item.get("threshold_op"),
        item.get("threshold_value"),
        item.get("threshold_value_min"),
        item.get("threshold_value_max"),
        item.get("threshold_unit"),
        item.get("disease_name"),
        item.get("section_type"),
        item.get("page"),
        _normalize_ascii(item.get("source_text") or "")[:120],
    )


def _make_threshold_id(item: dict[str, Any], index: int) -> str:
    disease = _slugify(item.get("disease_name") or "unknown")
    biomarker = _slugify(item.get("biomarker") or "value")
    op = {"<": "lt", ">": "gt", ">=": "gte", "<=": "lte", "between": "between"}.get(item.get("threshold_op"), "op")
    value = str(item.get("threshold_value") or "").replace(".", "_")
    page = item.get("page") or "x"
    return f"extra_{disease}_{biomarker}_{op}_{value}_p{page}_{index:03d}"


def _window(text: str, start: int, end: int, radius: int = 220) -> str:
    return " ".join(text[max(0, start - radius) : min(len(text), end + radius)].split())


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    return " ".join(normalized.encode("ascii", "ignore").decode("ascii").lower().split())


def _slugify(value: str) -> str:
    normalized = _normalize_ascii(value)
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "unknown"


if __name__ == "__main__":
    main()
