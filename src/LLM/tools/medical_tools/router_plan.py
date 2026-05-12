from __future__ import annotations

import json
from typing import Any

from src.LLM.tools.medical_tools.constants import (
    ALLOWED_DISEASE_NAMES,
    ALLOWED_SECTION_TYPES,
    ALLOWED_SOURCE_TYPES,
    DISEASE_NAME_ALIASES,
)
from src.LLM.tools.medical_tools.request_builder import sanitize_tool_parameters
from src.LLM.tools.medical_tools.text_utils import normalize_for_match


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


def normalize_router_plan(
    plan: dict[str, Any],
    query: str,
    *,
    extracted_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply safe defaults to router plan."""

    needs_tool = bool(plan.get("needs_medical_tool"))
    tool_call = plan.get("tool_call") if needs_tool else None
    if tool_call:
        endpoint = tool_call.get("endpoint") or "/mcp/medical-tools/evaluate"
        parameters = sanitize_tool_parameters(
            tool_call.get("parameters") or {},
            query=query,
            extracted_payload=extracted_payload,
        )
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
    filters["disease_name"] = canonical_disease_name(filters.get("disease_name"))
    section_type = optional_str(filters.get("section_type"))
    source_type = optional_str(filters.get("source_type")) or "chunk"
    filters["section_type"] = section_type if section_type in ALLOWED_SECTION_TYPES else None
    filters["source_type"] = source_type if source_type in ALLOWED_SOURCE_TYPES else "chunk"
    filters["biomarker"] = optional_str(filters.get("biomarker"))
    rag_plan["filters"] = filters

    return {
        "needs_medical_tool": needs_tool,
        "tool_call": tool_call,
        "rag_plan": rag_plan,
        "missing_inputs": plan.get("missing_inputs") or [],
        "reason": plan.get("reason"),
    }


def optional_str(value: Any) -> str | None:
    """Coerce router-provided filter to optional string."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list):
        for item in value:
            coerced = optional_str(item)
            if coerced:
                return coerced
    return None


def canonical_disease_name(value: Any) -> str | None:
    text = optional_str(value)
    if not text:
        return None
    if text in ALLOWED_DISEASE_NAMES:
        return text
    normalized = normalize_for_match(text)
    normalized_key = normalized.replace(" ", "_")
    if normalized_key in ALLOWED_DISEASE_NAMES:
        return normalized_key
    if text in DISEASE_NAME_ALIASES:
        return DISEASE_NAME_ALIASES[text]
    if normalized in DISEASE_NAME_ALIASES:
        return DISEASE_NAME_ALIASES[normalized]
    if normalized_key in DISEASE_NAME_ALIASES:
        return DISEASE_NAME_ALIASES[normalized_key]
    return None
