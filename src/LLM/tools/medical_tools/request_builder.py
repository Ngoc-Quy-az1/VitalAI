from __future__ import annotations

import re
from typing import Any

from services.medical_tools.aliases import BIOMARKER_ALIASES, FORMULA_VARIABLE_ALIASES, UNIT_ALIASES
from src.LLM.tools.medical_tools.constants import ALLOWED_DISEASE_NAMES, DISEASE_NAME_ALIASES
from src.LLM.tools.medical_tools.text_utils import normalize_for_match


NUMBER_RE = r"-?\d+(?:[.,]\d+)?"
UNIT_RE = r"g/24\s*(?:giờ|gio|h)|g/l|g/dl|mmol/l|mg/g|mg/mmol|mg/dl|μmol/l|µmol/l|umol/l|ml/ph(?:út|ut)?/1[.,]73m2|ml/ph(?:út|ut)?|mmhg|kg|cm|m2|%"
SUPPORTED_MEASUREMENT_NAMES = set(BIOMARKER_ALIASES) | set(FORMULA_VARIABLE_ALIASES)
SUPPORTED_FORMULA_IDS = {"mdrd_gfr", "cockcroft_gault", "body_surface_area", "fena_formula"}


def build_tool_input_payload(query: str) -> dict[str, Any]:
    """Extract a sanitized MCP payload candidate from user query.

    The output intentionally omits absent fields instead of sending nulls.
    """

    cleaned_query = " ".join((query or "").split())
    measurements = extract_supported_measurements(cleaned_query)
    payload: dict[str, Any] = {"text": cleaned_query}
    if measurements:
        payload["measurements"] = measurements
    disease_name = detect_disease_name_hint(cleaned_query)
    if disease_name:
        payload["disease_name"] = disease_name
    return payload


def sanitize_tool_parameters(
    parameters: dict[str, Any] | None,
    *,
    query: str,
    extracted_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge router parameters with deterministic extraction and drop nulls."""

    base = dict(extracted_payload or {})
    raw = parameters if isinstance(parameters, dict) else {}

    text = _clean_non_empty_string(raw.get("text")) or _clean_non_empty_string(base.get("text")) or " ".join(query.split())
    measurements = _merge_measurements(
        sanitize_measurements(base.get("measurements")),
        sanitize_measurements(raw.get("measurements")),
    )
    disease_name = canonical_disease_name(raw.get("disease_name")) or canonical_disease_name(base.get("disease_name"))
    formula_ids = sanitize_formula_ids(raw.get("formula_ids"))
    if not formula_ids and isinstance(base.get("formula_ids"), list):
        formula_ids = sanitize_formula_ids(base.get("formula_ids"))

    normalized: dict[str, Any] = {
        "text": text,
        "formula_ids": formula_ids,
        "include_debug": False,
    }
    if measurements:
        normalized["measurements"] = measurements
    if disease_name:
        normalized["disease_name"] = disease_name
    return normalized


def sanitize_measurements(value: Any) -> dict[str, dict[str, Any]]:
    """Keep only measurements/categorical variables supported by medical tools."""

    items: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                raw_name = item.get("name") or item.get("biomarker") or item.get("variable")
                items.append((str(raw_name or ""), item))

    sanitized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_value in items:
        name = canonical_measurement_name(raw_name)
        if not name:
            continue
        if name in {"sex", "race"}:
            categorical = _sanitize_categorical(name, raw_value)
            if categorical is not None:
                sanitized[name] = {"value": categorical}
            continue

        normalized_item = _sanitize_numeric_measurement(raw_value)
        if normalized_item is not None:
            sanitized[name] = normalized_item

    _enrich_formula_measurements(sanitized)
    return sanitized


def extract_supported_measurements(query: str) -> dict[str, dict[str, Any]]:
    """Deterministically parse supported measurements/variables from text."""

    measurements: dict[str, dict[str, Any]] = {}
    normalized = normalize_for_match(query)

    sex = _detect_sex(normalized)
    race = _detect_race(normalized)
    if sex:
        measurements["sex"] = {"value": sex}
    if race:
        measurements["race"] = {"value": race}

    for name, aliases in BIOMARKER_ALIASES.items():
        parsed = _extract_value_for_aliases(query, aliases)
        if parsed:
            measurements[name] = _build_numeric_item(parsed[0], parsed[1])

    for name, aliases in FORMULA_VARIABLE_ALIASES.items():
        if name in {"sex", "race"}:
            continue
        parsed = _extract_value_for_aliases(query, aliases)
        if parsed:
            measurements[name] = _build_numeric_item(parsed[0], parsed[1])

    _enrich_formula_measurements(measurements)
    return measurements


def tool_payload_has_supported_inputs(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    measurements = payload.get("measurements")
    return isinstance(measurements, dict) and bool(measurements)


def build_supported_tool_context() -> str:
    biomarker_names = ", ".join(sorted(BIOMARKER_ALIASES))
    variable_names = ", ".join(sorted(FORMULA_VARIABLE_ALIASES))
    formula_ids = ", ".join(sorted(SUPPORTED_FORMULA_IDS))
    return (
        f"Supported biomarkers: {biomarker_names}\n"
        f"Supported formula variables: {variable_names}\n"
        f"Supported formula_ids: {formula_ids}\n"
        "Rule: chỉ dùng đúng field có trong danh sách trên; field nào không chắc hoặc không có giá trị thì bỏ hẳn, không ghi null."
    )


def sanitize_formula_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_non_empty_string(item)
        if not text or text not in SUPPORTED_FORMULA_IDS or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def canonical_measurement_name(value: Any) -> str | None:
    text = _clean_non_empty_string(value)
    if not text:
        return None
    normalized = normalize_for_match(text)
    for canonical in SUPPORTED_MEASUREMENT_NAMES:
        if normalize_for_match(canonical) == normalized:
            return canonical
    for canonical, aliases in BIOMARKER_ALIASES.items():
        if any(normalize_for_match(alias) == normalized for alias in [canonical, *aliases]):
            return canonical
    for canonical, aliases in FORMULA_VARIABLE_ALIASES.items():
        if any(normalize_for_match(alias) == normalized for alias in [canonical, *aliases]):
            return canonical
    return None


def detect_disease_name_hint(query: str) -> str | None:
    normalized = normalize_for_match(query)
    normalized_key = normalized.replace(" ", "_")
    if normalized_key in ALLOWED_DISEASE_NAMES:
        return normalized_key
    if normalized in DISEASE_NAME_ALIASES:
        return DISEASE_NAME_ALIASES[normalized]
    if normalized_key in DISEASE_NAME_ALIASES:
        return DISEASE_NAME_ALIASES[normalized_key]
    for alias, canonical in DISEASE_NAME_ALIASES.items():
        alias_normalized = normalize_for_match(alias)
        if alias_normalized and alias_normalized in normalized:
            return canonical
    return None


def canonical_disease_name(value: Any) -> str | None:
    text = _clean_non_empty_string(value)
    if not text:
        return None
    normalized = normalize_for_match(text)
    normalized_key = normalized.replace(" ", "_")
    if text in ALLOWED_DISEASE_NAMES:
        return text
    if normalized_key in ALLOWED_DISEASE_NAMES:
        return normalized_key
    if text in DISEASE_NAME_ALIASES:
        return DISEASE_NAME_ALIASES[text]
    if normalized in DISEASE_NAME_ALIASES:
        return DISEASE_NAME_ALIASES[normalized]
    if normalized_key in DISEASE_NAME_ALIASES:
        return DISEASE_NAME_ALIASES[normalized_key]
    return None


def _merge_measurements(*sources: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source in sources:
        for key, value in source.items():
            merged[key] = value
    return merged


def _sanitize_numeric_measurement(raw_value: Any) -> dict[str, Any] | None:
    item = raw_value if isinstance(raw_value, dict) else {"value": raw_value}
    value = _to_float(item.get("value"))
    if value is None:
        return None
    return _build_numeric_item(value, item.get("unit"))


def _sanitize_categorical(name: str, raw_value: Any) -> str | None:
    item = raw_value if isinstance(raw_value, dict) else {"value": raw_value}
    raw = _clean_non_empty_string(item.get("value"))
    if not raw:
        return None
    normalized = normalize_for_match(raw)
    if name == "sex":
        return _detect_sex(normalized)
    if name == "race":
        return _detect_race(normalized)
    return None


def _extract_value_for_aliases(text: str, aliases: list[str]) -> tuple[float, str | None] | None:
    for alias in sorted(aliases, key=len, reverse=True):
        alias_pattern = re.escape(alias).replace("\\ ", r"\s+")
        patterns = [
            rf"(?i)(?<!\w){alias_pattern}(?!\w)\s*(?:=|:|là|la)?\s*(?P<value>{NUMBER_RE})\s*(?P<unit>{UNIT_RE})?",
            rf"(?i)(?P<value>{NUMBER_RE})\s*(?P<unit>{UNIT_RE})?\s*(?<!\w){alias_pattern}(?!\w)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            value = _to_float(match.group("value"))
            if value is None:
                continue
            return value, _normalize_unit(match.groupdict().get("unit"))
    return None


def _enrich_formula_measurements(measurements: dict[str, dict[str, Any]]) -> None:
    creatinine = measurements.get("creatinine")
    if creatinine and "creatinine_mg_dl" not in measurements and creatinine.get("unit") in {None, "mg/dL"}:
        measurements["creatinine_mg_dl"] = {"value": creatinine["value"], "unit": "mg/dL"}
    plasma_creatinine = measurements.get("plasma_creatinine")
    if plasma_creatinine and "creatinine_mg_dl" not in measurements and plasma_creatinine.get("unit") in {None, "mg/dL"}:
        measurements["creatinine_mg_dl"] = {"value": plasma_creatinine["value"], "unit": "mg/dL"}


def _build_numeric_item(value: float, unit: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"value": value}
    normalized_unit = _normalize_unit(unit)
    if normalized_unit:
        result["unit"] = normalized_unit
    return result


def _detect_sex(normalized_text: str) -> str | None:
    if re.search(r"(?<![a-z0-9])(nu|female|woman)(?![a-z0-9])", normalized_text):
        return "female"
    if re.search(r"(?<![a-z0-9])(nam|male|man)(?![a-z0-9])", normalized_text):
        return "male"
    return None


def _detect_race(normalized_text: str) -> str | None:
    if re.search(r"(?<![a-z0-9])(black|da den|african)(?![a-z0-9])", normalized_text):
        return "black"
    if re.search(r"(?<![a-z0-9])(other|khac|chau a|asian)(?![a-z0-9])", normalized_text):
        return "other"
    return None


def _normalize_unit(unit: Any) -> str | None:
    if unit is None:
        return None
    value = str(unit).strip()
    if not value:
        return None
    normalized = value.replace(" ", "")
    return UNIT_ALIASES.get(value, UNIT_ALIASES.get(normalized, value))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return None


def _clean_non_empty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
