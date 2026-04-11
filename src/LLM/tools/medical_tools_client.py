from __future__ import annotations

"""HTTP client và helpers cho Medical Tools MCP endpoint."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "tool_contracts" / "medical_tools_contract.md"
ALLOWED_ENDPOINTS = {"/mcp/medical-tools/evaluate"}
ALLOWED_SECTION_TYPES = {
    "definition",
    "classification",
    "clinical_features",
    "diagnosis_criteria",
    "pathology",
    "treatment",
    "progression",
    "complications",
    "follow_up",
    "general",
}
ALLOWED_SOURCE_TYPES = {"chunk", "threshold", "formula"}


class MedicalToolsClient:
    """Minimal async wrapper around the deployable medical tools HTTP service."""

    def __init__(self, base_url: str, timeout_seconds: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds

    async def evaluate(self, parameters: dict[str, Any], endpoint: str = "/mcp/medical-tools/evaluate") -> dict[str, Any]:
        """Call medical tools evaluate endpoint and return parsed JSON.

        Endpoint path is validated against the contract allow-list so router output
        cannot redirect the graph to arbitrary URLs.
        """

        if endpoint not in ALLOWED_ENDPOINTS:
            return {
                "tool_status": "blocked",
                "error": f"Endpoint không được phép: {endpoint}",
            }
        return await asyncio.to_thread(self._post_json, endpoint, parameters)

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = urljoin(self.base_url, endpoint.lstrip("/"))
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            return {"tool_status": "http_error", "status_code": exc.code, "error": str(exc)}
        except URLError as exc:
            return {"tool_status": "unavailable", "error": str(exc.reason)}
        except TimeoutError:
            return {"tool_status": "timeout", "error": "Medical tools service timeout"}
        except json.JSONDecodeError as exc:
            return {"tool_status": "bad_json", "error": str(exc)}


def load_medical_tools_contract(path: str | Path | None = None) -> str:
    """Read runtime MCP contract markdown for the router agent."""

    contract_path = Path(path) if path else DEFAULT_CONTRACT_PATH
    return contract_path.read_text(encoding="utf-8")


def parse_router_plan(raw_content: str) -> dict[str, Any]:
    """Parse JSON object from router LLM output."""

    content = raw_content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Router output không chứa JSON object hợp lệ")
    plan = json.loads(content[start : end + 1])
    if not isinstance(plan, dict):
        raise ValueError("Router output phải là JSON object")
    return plan


def normalize_router_plan(plan: dict[str, Any], query: str) -> dict[str, Any]:
    """Apply safe defaults to router plan."""

    needs_tool = bool(plan.get("needs_medical_tool"))
    tool_call = plan.get("tool_call") if needs_tool else None
    if tool_call:
        endpoint = tool_call.get("endpoint") or "/mcp/medical-tools/evaluate"
        parameters = tool_call.get("parameters") or {}
        if not isinstance(parameters, dict):
            parameters = {}
        parameters.setdefault("text", query)
        parameters.setdefault("measurements", None)
        parameters.setdefault("disease_name", None)
        parameters.setdefault("formula_ids", [])
        if parameters.get("formula_ids") is None:
            # Không tính mọi công thức theo default; chỉ tính khi router chọn
            # formula_ids cụ thể để tránh noise và giảm token ở final prompt.
            parameters["formula_ids"] = []
        parameters["include_debug"] = False
        tool_call = {
            "tool_name": "medical_tools.evaluate",
            "method": "POST",
            "endpoint": endpoint,
            "parameters": parameters,
        }

    rag_plan = plan.get("rag_plan") or {}
    if not isinstance(rag_plan, dict):
        rag_plan = {}
    rag_plan.setdefault("should_retrieve", True)
    rag_plan.setdefault("query", query)
    filters = rag_plan.get("filters") or {}
    if not isinstance(filters, dict):
        filters = {}
    filters["disease_name"] = _optional_str(filters.get("disease_name"))
    section_type = _optional_str(filters.get("section_type"))
    source_type = _optional_str(filters.get("source_type")) or "chunk"
    filters["section_type"] = section_type if section_type in ALLOWED_SECTION_TYPES else None
    filters["source_type"] = source_type if source_type in ALLOWED_SOURCE_TYPES else "chunk"
    filters["biomarker"] = _optional_str(filters.get("biomarker"))
    rag_plan["filters"] = filters

    return {
        "needs_medical_tool": needs_tool,
        "tool_call": tool_call,
        "rag_plan": rag_plan,
        "missing_inputs": plan.get("missing_inputs") or [],
        "reason": plan.get("reason"),
    }


def build_structured_context(result: dict[str, Any] | None, query: str | None = None) -> str:
    """Format medical tools JSON into compact, sanitized context for final prompt.

    Final answer generation should never receive raw tool JSON because it is
    token-heavy and contains internal fields such as threshold_id/source_text.
    This formatter keeps only user-relevant clinical facts.
    """

    if not result:
        return "Không có kết quả phân tích chỉ số."
    if result.get("tool_status") in {"unavailable", "timeout", "http_error", "bad_json", "blocked"}:
        return (
            "Kết quả phân tích chỉ số: không khả dụng. "
            "Không được tự tính công thức hoặc tự phân loại ngưỡng nếu thiếu dữ liệu chắc chắn."
        )

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
    disclaimer = _clean_text(safety.get("medical_disclaimer"))
    if disclaimer:
        lines.append(f"- Lưu ý an toàn: {disclaimer}")

    if len(lines) == 1:
        return "Không có kết quả threshold, phân loại hoặc công thức phù hợp từ chỉ số đã cung cấp."
    return "\n".join(lines)


def build_structured_answer(result: dict[str, Any] | None, query: str | None = None) -> str | None:
    """Build a safe user-facing answer when structured tool result is enough.

    This bypasses final LLM synthesis for tool-only cases, preventing the model
    from adding unsupported symptoms, causes, treatments, or broad explanations.
    """

    if not result:
        return None
    if result.get("tool_status") in {"unavailable", "timeout", "http_error", "bad_json", "blocked"}:
        return (
            "Hiện chưa có đủ dữ liệu phân tích chỉ số để trả lời chắc chắn. "
            "Không nên tự suy diễn kết quả công thức hoặc phân loại khi dữ liệu chưa khả dụng."
        )

    measurements = _format_measurements(result.get("detected_measurements", []))
    threshold_lines = _matched_threshold_lines(result.get("threshold_matches", []), include_classifications=False)
    classification_lines = _matched_threshold_lines(result.get("classifications", []), include_classifications=True)
    formula_lines = _formula_result_lines(result.get("formula_results", []), query=query)

    detail_lines = [*threshold_lines[:6], *classification_lines[:6], *formula_lines[:4]]
    if not detail_lines:
        return None

    opening = "Các chỉ số bạn cung cấp có điểm cần chú ý."
    if measurements:
        opening = f"Các chỉ số bạn cung cấp ({measurements}) có điểm cần chú ý."

    answer_lines = [opening, ""]
    answer_lines.extend(f"- {line}" for line in detail_lines)
    answer_lines.extend(
        [
            "",
            "Kết quả này chỉ mang tính tham khảo và không thay thế đánh giá của bác sĩ.",
        ]
    )
    return "\n".join(answer_lines)


def _limit_items(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:limit]


def _format_measurements(items: Any) -> str:
    if not isinstance(items, list):
        return ""

    parts: list[str] = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        name = _clean_text(item.get("name"))
        value = _format_number(item.get("value"))
        unit = _clean_text(item.get("unit"))
        if not name or value is None:
            continue
        parts.append(_join_value_unit(name, value, unit))
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
        label = _clean_text(threshold.get("label"))
        if include_classifications and not label:
            continue
        if not include_classifications and label:
            continue

        biomarker = _clean_text(item.get("biomarker"))
        input_value = _format_number(item.get("input_value"))
        input_unit = _clean_text(item.get("input_unit"))
        comparison_unit = _clean_text(item.get("comparison_unit"))
        op_text = _threshold_text(threshold)
        if not biomarker or input_value is None or not op_text:
            continue

        key = (biomarker, input_value, comparison_unit, label or op_text)
        if key in seen:
            continue
        seen.add(key)

        value_text = _join_value_unit(biomarker, input_value, input_unit or comparison_unit)
        if label:
            lines.append(f"{value_text} thuộc {label} ({op_text}).")
        else:
            lines.append(f"{value_text} thỏa điều kiện {op_text}.")
    return lines


def _formula_result_lines(items: Any, *, query: str | None) -> list[str]:
    if not isinstance(items, list):
        return []

    query_norm = _normalize_for_match(query or "")
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        formula_name = _clean_text(item.get("formula_name")) or _clean_text(item.get("formula_id")) or "Công thức"
        if status == "computed":
            value = _format_number(item.get("value"))
            unit = _clean_text(item.get("unit"))
            if value is not None:
                lines.append(f"{formula_name}: {value}{f' {unit}' if unit else ''}.")
            continue

        if status == "missing_inputs" and _formula_is_relevant(item, query_norm):
            missing = [_clean_text(value) for value in item.get("missing_inputs", []) if _clean_text(value)]
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
        normalized = _normalize_for_match(str(candidate or ""))
        if normalized and normalized in query_norm:
            return True
    return any(keyword in query_norm for keyword in ("cong thuc", "tinh egfr", "tinh gfr", "mdrd", "cockcroft", "bsa", "fena"))


def _threshold_text(threshold: dict[str, Any]) -> str:
    op = threshold.get("op")
    unit = _clean_text(threshold.get("unit"))
    if op == "between":
        value_min = _format_number(threshold.get("value_min"))
        value_max = _format_number(threshold.get("value_max"))
        if value_min is None or value_max is None:
            return ""
        return f"từ {value_min} đến dưới {value_max}{f' {unit}' if unit else ''}"

    value = _format_number(threshold.get("value"))
    if value is None:
        return ""
    op_map = {
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
    }
    return f"{op_map.get(str(op), str(op))} {value}{f' {unit}' if unit else ''}"


def _join_value_unit(name: str, value: str, unit: str | None) -> str:
    return f"{name} {value}{f' {unit}' if unit else ''}"


def _format_number(value: Any) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number.is_integer():
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_for_match(value: str) -> str:
    normalized = value.lower()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^\w\s]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _optional_str(value: Any) -> str | None:
    """Coerce router-provided filter to optional string.

    LLM routers sometimes emit a list for fields such as biomarker. The retriever
    currently accepts only one value, so we choose the first non-empty string.
    """

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list):
        for item in value:
            coerced = _optional_str(item)
            if coerced:
                return coerced
    return None
