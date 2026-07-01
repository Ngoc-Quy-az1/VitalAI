from __future__ import annotations

from typing import Any
import json

from src.LLM.tools.medical_tools.constants import LABEL_TRANSLATIONS
from src.LLM.tools.medical_tools.text_utils import clean_text, format_number, join_value_unit, normalize_for_match


def build_structured_context(result: dict[str, Any] | None, query: str | None = None) -> str:
    """Format medical tools JSON into compact, sanitized context for final prompt."""

    if not result:
        return "Không có kết quả phân tích chỉ số."
    if result.get("tool_status") in {"unavailable", "timeout", "http_error", "bad_json", "blocked"}:
        return (
            "Kết quả phân tích chỉ số: không khả dụng. "
            "Không được tự tính công thức hoặc tự phân loại ngưỡng nếu thiếu dữ liệu chắc chắn."
        )

    if result.get("result_type") == "structured_knowledge_query":
        hits = result.get("hits") or []
        if not hits:
            return "Không tìm thấy sơ đồ hoặc bảng dữ liệu có cấu trúc phù hợp."
        
        lines: list[str] = ["Dữ liệu sơ đồ/bảng có cấu trúc tìm được:"]
        for hit in hits:
            doc_id = hit.get("document_id")
            source_type = hit.get("source_type")
            content = hit.get("content") or {}
            
            content_str = json.dumps(content, ensure_ascii=False, indent=2)
            lines.append(f"\n--- [ID: {doc_id}, Type: {source_type}] ---\n{content_str}")
        return "\n".join(lines)

    lines: list[str] = ["Kết quả phân tích chỉ số đã được chuẩn hóa:"]

    measurements = _format_measurements(result.get("detected_measurements", []))
    if measurements:
        lines.append(f"- Chỉ số nhận diện: {measurements}.")

    derived = _format_measurements(result.get("derived_measurements", []))
    if derived:
        lines.append(f"- Chỉ số tính từ công thức: {derived}.")

    threshold_lines = _matched_threshold_lines(result.get("threshold_matches", []), include_classifications=False)
    if threshold_lines:
        lines.append("- Ngưỡng khớp:")
        lines.extend(f"  - {line}" for line in threshold_lines[:6])

    classification_lines = _matched_threshold_lines(result.get("classifications", []), include_classifications=True)
    if classification_lines:
        lines.append("- Phân loại:")
        lines.extend(f"  - {line}" for line in classification_lines[:6])

    formula_lines = _formula_result_lines(result.get("formula_results", []), query=query)
    if formula_lines:
        lines.append("- Công thức:")
        lines.extend(f"  - {line}" for line in formula_lines[:4])

    safety = result.get("safety") or {}
    disclaimer = clean_text(safety.get("medical_disclaimer"))
    if disclaimer:
        lines.append(f"- Lưu ý an toàn: {disclaimer}")

    if len(lines) == 1:
        return "Không có kết quả threshold, phân loại hoặc công thức phù hợp từ chỉ số đã cung cấp."
    return "\n".join(lines)


def _format_measurements(items: Any) -> str:
    if not isinstance(items, list):
        return ""

    parts: list[str] = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        name = clean_text(item.get("name"))
        value = format_number(item.get("value"))
        unit = clean_text(item.get("unit"))
        if not name or value is None:
            continue
        parts.append(join_value_unit(name, value, unit))
    return "; ".join(parts)


def _matched_threshold_lines(items: Any, *, include_classifications: bool) -> list[str]:
    if not isinstance(items, list):
        return []

    seen: set[tuple[str, str, str, str]] = set()
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("matched"):
            continue

        threshold = item.get("threshold") or {}
        label = _display_label(threshold.get("label"))
        if include_classifications and not label:
            continue
        if not include_classifications and label:
            continue

        biomarker = clean_text(item.get("biomarker"))
        input_value = format_number(item.get("input_value"))
        input_unit = clean_text(item.get("input_unit"))
        comparison_unit = clean_text(item.get("comparison_unit"))
        op_text = _threshold_text(threshold)
        if not biomarker or input_value is None or not op_text:
            continue

        key = (biomarker, input_value, comparison_unit, label or op_text)
        if key in seen:
            continue
        seen.add(key)

        value_text = join_value_unit(biomarker, input_value, input_unit or comparison_unit)
        if label:
            lines.append(f"{value_text} thuộc {label} ({op_text}).")
        else:
            lines.append(f"{value_text} thỏa điều kiện {op_text}.")
    return lines


def _formula_result_lines(items: Any, *, query: str | None) -> list[str]:
    if not isinstance(items, list):
        return []

    query_norm = normalize_for_match(query or "")
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        formula_name = clean_text(item.get("formula_name")) or clean_text(item.get("formula_id")) or "Công thức"
        if status == "computed":
            value = format_number(item.get("value"))
            unit = clean_text(item.get("unit")) or _default_formula_unit(item)
            output_label = _formula_output_label(item)
            assumption_text = _formula_assumption_text(item)
            if value is not None:
                line = f"{formula_name} tính {output_label}: {value}{f' {unit}' if unit else ''}."
                if assumption_text:
                    line = f"{line} {assumption_text}"
                lines.append(line)
            continue

        if status == "missing_inputs" and _formula_is_relevant(item, query_norm):
            missing = [clean_text(value) for value in item.get("missing_inputs", []) if clean_text(value)]
            if missing:
                lines.append(f"{formula_name} chưa tính được vì thiếu: {', '.join(missing[:6])}.")
    return lines


def _formula_is_relevant(item: dict[str, Any], query_norm: str) -> bool:
    if not query_norm:
        return False
    candidates = [
        item.get("formula_id"),
        item.get("formula_name"),
        item.get("output_name"),
    ]
    for candidate in candidates:
        normalized = normalize_for_match(str(candidate or ""))
        if normalized and normalized in query_norm:
            return True
    return any(keyword in query_norm for keyword in ("cong thuc", "tinh egfr", "tinh gfr", "mdrd", "cockcroft", "bsa", "fena"))


def _threshold_text(threshold: dict[str, Any]) -> str:
    op = threshold.get("op")
    unit = clean_text(threshold.get("unit"))
    if op == "between":
        value_min = format_number(threshold.get("value_min"))
        value_max = format_number(threshold.get("value_max"))
        if value_min is None or value_max is None:
            return ""
        return f"từ {value_min} đến dưới {value_max}{f' {unit}' if unit else ''}"

    value = format_number(threshold.get("value"))
    if value is None:
        return ""
    op_map = {
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
    }
    return f"{op_map.get(str(op), str(op))} {value}{f' {unit}' if unit else ''}"


def _display_label(value: Any) -> str:
    label = clean_text(value)
    return LABEL_TRANSLATIONS.get(label, label)


def _default_formula_unit(item: dict[str, Any]) -> str:
    if item.get("output_name") == "fena_percent":
        return "%"
    return ""


def _formula_output_label(item: dict[str, Any]) -> str:
    output_name = clean_text(item.get("output_name"))
    labels = {
        "fena_percent": "FENa",
        "gfr_ml_min_1_73m2": "eGFR",
        "creatinine_clearance_ml_min": "độ thanh thải creatinine",
        "bsa_m2": "diện tích da cơ thể",
    }
    return labels.get(output_name, output_name or "kết quả")


def _formula_assumption_text(item: dict[str, Any]) -> str:
    assumptions = [clean_text(value) for value in item.get("assumptions", []) if clean_text(value)]
    if not assumptions:
        return ""
    return f"Lưu ý: {' '.join(assumptions[:2])}"
