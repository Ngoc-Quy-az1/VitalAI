from __future__ import annotations

"""Compatibility facade for the refactored medical tools client package."""

from src.LLM.tools.medical_tools import (
    MedicalToolsClient,
    build_structured_context,
    load_medical_tools_contract,
    normalize_router_plan,
    parse_router_plan,
)

__all__ = [
    "MedicalToolsClient",
    "build_structured_context",
    "load_medical_tools_contract",
    "normalize_router_plan",
    "parse_router_plan",
]
