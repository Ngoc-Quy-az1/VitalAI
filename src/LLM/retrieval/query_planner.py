from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.LLM.retrieval.vector_search import (
    BIOMARKER_HINTS,
    DISEASE_HINTS,
    DISEASE_LABELS,
    SECTION_HINTS,
    SECTION_LABELS,
)
from src.LLM.tools.medical_tools.constants import (
    ALLOWED_DISEASE_NAMES,
    ALLOWED_SECTION_TYPES,
    ALLOWED_SOURCE_TYPES,
    DISEASE_NAME_ALIASES,
)


HARD_DISEASE_CONFIDENCE = 0.86
HARD_DISEASE_MARGIN = 0.12
SOFT_ONLY_DISEASES = {"benh_ly_cau_than", "viem_cau_than_man"}
SOFT_ONLY_BIOMARKERS = {"FENa"}
ALLOWED_RETRIEVAL_DISEASE_NAMES = set(DISEASE_HINTS) | set(ALLOWED_DISEASE_NAMES)


def build_retrieval_plan(
    *,
    query: str,
    initial_filters: dict[str, Any] | None = None,
    router_plan: dict[str, Any] | None = None,
    extracted_tool_payload: dict[str, Any] | None = None,
    medical_tool_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a safe, tool-aware plan for RAG retrieval."""

    initial_filters = initial_filters or {}
    router_plan = router_plan or {}
    rag_plan = router_plan.get("rag_plan") if isinstance(router_plan.get("rag_plan"), dict) else {}
    router_filters = rag_plan.get("filters") if isinstance(rag_plan.get("filters"), dict) else {}
    extracted_tool_payload = extracted_tool_payload or {}

    base_query = _clean_text(rag_plan.get("query")) or query
    normalized_query = _normalize_text(query)
    candidates = {
        "diseases": _merge_candidates(
            _detect_candidates(normalized_query, DISEASE_HINTS, kind="disease"),
            _candidate_from_value(
                extracted_tool_payload.get("disease_name"),
                confidence=0.9,
                source="deterministic_payload",
            ),
            _candidate_from_value(
                router_filters.get("disease_name"),
                confidence=0.78,
                source="medical_router_rag_plan",
            ),
            *_tool_disease_candidates(medical_tool_result),
        ),
        "sections": _merge_candidates(
            _detect_candidates(normalized_query, SECTION_HINTS, kind="section"),
            _candidate_from_value(
                router_filters.get("section_type"),
                confidence=0.74,
                source="medical_router_rag_plan",
            ),
        ),
        "biomarkers": _merge_candidates(
            _detect_candidates(normalized_query, BIOMARKER_HINTS, kind="biomarker"),
            *_payload_biomarker_candidates(extracted_tool_payload),
            *_tool_biomarker_candidates(medical_tool_result),
            _candidate_from_value(
                router_filters.get("biomarker"),
                confidence=0.76,
                source="medical_router_rag_plan",
            ),
        ),
    }

    selected_disease = _select_top(candidates["diseases"])
    hard_disease = _select_hard_disease(candidates["diseases"])
    selected_section = _select_top(candidates["sections"])
    selected_biomarker = _select_top(candidates["biomarkers"])
    query_type = _query_type(query=query, router_plan=router_plan, extracted_tool_payload=extracted_tool_payload)
    tool_query_parts = _tool_query_parts(medical_tool_result)

    hard_filters = {
        "disease_name": _explicit_filter(initial_filters.get("disease_name")) or hard_disease,
        "section_type": _explicit_section_filter(initial_filters.get("section_type")),
        "source_type": _explicit_source_filter(initial_filters.get("source_type"))
        or _explicit_source_filter(router_filters.get("source_type"))
        or "chunk",
        "biomarker": _explicit_filter(initial_filters.get("biomarker")),
    }

    if hard_filters["biomarker"] in SOFT_ONLY_BIOMARKERS:
        hard_filters["biomarker"] = None

    soft_hints = {
        "disease_names": [item["name"] for item in candidates["diseases"] if item["confidence"] >= 0.55],
        "section_types": [item["name"] for item in candidates["sections"] if item["confidence"] >= 0.55],
        "biomarkers": [item["name"] for item in candidates["biomarkers"] if item["confidence"] >= 0.55],
        "terms": _key_terms(query=query, candidates=candidates, tool_query_parts=tool_query_parts),
    }

    retrieval_query = _build_query(
        base_query=base_query,
        tool_query_parts=tool_query_parts,
        disease_name=hard_filters["disease_name"] or selected_disease,
        section_type=selected_section,
        biomarker=selected_biomarker,
        soft_hints=soft_hints,
    )

    return {
        "strategy": "deterministic_tool_aware_query_planner_v1",
        "query_type": query_type,
        "query": retrieval_query,
        "filters": hard_filters,
        "soft_hints": soft_hints,
        "candidates": candidates,
        "confidence": {
            "disease": _confidence(selected_disease, candidates["diseases"]),
            "section": _confidence(selected_section, candidates["sections"]),
            "biomarker": _confidence(selected_biomarker, candidates["biomarkers"]),
        },
        "reason": _reason(query_type=query_type, filters=hard_filters, candidates=candidates, tool_result=medical_tool_result),
    }


def _detect_candidates(normalized_query: str, hint_map: dict[str, list[str]], *, kind: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for name, aliases in hint_map.items():
        for alias in aliases:
            normalized_alias = _normalize_text(alias)
            if not normalized_alias or not _contains_term(normalized_query, normalized_alias):
                continue
            candidates.append(
                {
                    "name": name,
                    "confidence": _alias_confidence(normalized_alias, kind=kind),
                    "source": "query_alias",
                    "evidence_terms": [alias],
                }
            )
    return candidates


def _candidate_from_value(value: Any, *, confidence: float, source: str) -> list[dict[str, Any]]:
    text = _clean_text(value)
    if not text:
        return []
    canonical = _canonical_disease_name(text) if source != "medical_router_rag_plan" else text
    if canonical in ALLOWED_RETRIEVAL_DISEASE_NAMES:
        return [{"name": canonical, "confidence": confidence, "source": source, "evidence_terms": [text]}]
    if text in ALLOWED_SECTION_TYPES or text in BIOMARKER_HINTS:
        return [{"name": text, "confidence": confidence, "source": source, "evidence_terms": [text]}]
    return []


def _payload_biomarker_candidates(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    measurements = payload.get("measurements")
    if not isinstance(measurements, dict):
        return []
    return [
        [{"name": name, "confidence": 0.86, "source": "deterministic_payload", "evidence_terms": [name]}]
        for name in measurements
        if name in BIOMARKER_HINTS
    ]


def _tool_disease_candidates(tool_result: dict[str, Any] | None) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for item in _matched_threshold_items(tool_result):
        disease_name = (item.get("threshold") or {}).get("disease_name")
        canonical = _canonical_disease_name(disease_name)
        if canonical:
            result.append(
                [
                    {
                        "name": canonical,
                        "confidence": 0.96,
                        "source": "medical_tool_result",
                        "evidence_terms": [canonical],
                    }
                ]
            )
    return result


def _tool_biomarker_candidates(tool_result: dict[str, Any] | None) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for item in _matched_threshold_items(tool_result):
        biomarker = _clean_text(item.get("biomarker"))
        if biomarker:
            result.append(
                [
                    {
                        "name": biomarker,
                        "confidence": 0.9,
                        "source": "medical_tool_result",
                        "evidence_terms": [biomarker],
                    }
                ]
            )
    for item in (tool_result or {}).get("derived_measurements", []) or []:
        if isinstance(item, dict) and _clean_text(item.get("name")):
            name = _clean_text(item.get("name"))
            result.append(
                [
                    {
                        "name": name,
                        "confidence": 0.82,
                        "source": "medical_tool_result",
                        "evidence_terms": [name],
                    }
                ]
            )
    return result


def _matched_threshold_items(tool_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(tool_result, dict) or tool_result.get("tool_status"):
        return []
    return [
        item
        for item in tool_result.get("threshold_matches", []) or []
        if isinstance(item, dict) and item.get("matched")
    ]


def _tool_query_parts(tool_result: dict[str, Any] | None) -> list[str]:
    if not isinstance(tool_result, dict) or tool_result.get("tool_status"):
        return []
    parts: list[str] = []
    formulas = _unique(
        _clean_text(item.get("formula_name") or item.get("formula_id"))
        for item in tool_result.get("formula_results", []) or []
        if isinstance(item, dict)
    )
    labels = _unique(
        _clean_text((item.get("threshold") or {}).get("label"))
        for item in _matched_threshold_items(tool_result)
    )
    source_texts = _unique(
        _clean_text((item.get("source") or {}).get("source_text"))
        for item in _matched_threshold_items(tool_result)
    )
    if formulas:
        parts.append("Công thức: " + ", ".join(formulas[:3]))
    if labels:
        parts.append("Ý nghĩa lâm sàng: " + ", ".join(labels[:5]))
    if source_texts:
        parts.append("Ngữ cảnh ngưỡng: " + " ".join(source_texts[:2]))
    return parts


def _select_hard_disease(candidates: list[dict[str, Any]]) -> str | None:
    """Choose a DB disease filter only from tool-grounded evidence.

    Query aliases are intentionally soft-only. The current corpus has many
    useful chunks under broad metadata such as `benh_ly_cau_than`, so hard
    filtering from query text can hurt recall and groundedness.
    """

    ranked = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
    if not ranked:
        return None
    top = ranked[0]
    runner_up = ranked[1]["confidence"] if len(ranked) > 1 else 0.0
    if "medical_tool_result" not in str(top.get("source") or "").split("+"):
        return None
    if top["name"] in SOFT_ONLY_DISEASES:
        return None
    if top["confidence"] < HARD_DISEASE_CONFIDENCE:
        return None
    if runner_up and top["confidence"] - runner_up < HARD_DISEASE_MARGIN:
        return None
    return top["name"]


def _select_top(candidates: list[dict[str, Any]]) -> str | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["confidence"])["name"]


def _merge_candidates(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for item in group:
            name = _clean_text(item.get("name"))
            if not name:
                continue
            existing = merged.setdefault(
                name,
                {
                    "name": name,
                    "confidence": 0.0,
                    "source": "",
                    "evidence_terms": [],
                },
            )
            existing["confidence"] = round(max(existing["confidence"], float(item.get("confidence") or 0.0)), 3)
            source = _clean_text(item.get("source"))
            if source and source not in existing["source"].split("+"):
                existing["source"] = "+".join(filter(None, [existing["source"], source]))
            existing["evidence_terms"] = _unique(
                [*existing["evidence_terms"], *[str(term) for term in item.get("evidence_terms", []) if term]]
            )
    return sorted(merged.values(), key=lambda item: item["confidence"], reverse=True)


def _build_query(
    *,
    base_query: str,
    tool_query_parts: list[str],
    disease_name: str | None,
    section_type: str | None,
    biomarker: str | None,
    soft_hints: dict[str, list[str]],
) -> str:
    parts = [_clean_text(base_query)]
    parts.extend(tool_query_parts)
    if disease_name:
        parts.append(f"Bệnh trọng tâm: {DISEASE_LABELS.get(disease_name, disease_name)}")
    if section_type:
        parts.append(f"Mục cần tìm: {SECTION_LABELS.get(section_type, section_type)}")
    if biomarker:
        parts.append(f"Chỉ số trọng tâm: {biomarker}")
    if soft_hints.get("terms"):
        parts.append("Từ khóa ưu tiên: " + ", ".join(soft_hints["terms"][:12]))
    return "\n".join(_unique(part for part in parts if part))


def _key_terms(*, query: str, candidates: dict[str, list[dict[str, Any]]], tool_query_parts: list[str]) -> list[str]:
    normalized_query = _normalize_text(query)
    tokens = re.findall(r"[a-z0-9]+", normalized_query)
    acronyms = [token for token in tokens if token in {"acr", "pcr", "gfr", "egfr", "aki", "ckd", "kdigo", "kdoqi", "rifle", "ara", "fena", "aslo"}]
    numbers = re.findall(r"\d+(?:[\.,]\d+)?", normalized_query)
    evidence_terms: list[str] = []
    for group in candidates.values():
        for item in group:
            evidence_terms.extend(_normalize_text(term) for term in item.get("evidence_terms", []))
    for part in tool_query_parts:
        evidence_terms.extend(re.findall(r"[a-z0-9_]{3,}", _normalize_text(part))[:8])
    return _unique([*acronyms, *numbers, *evidence_terms])


def _query_type(*, query: str, router_plan: dict[str, Any], extracted_tool_payload: dict[str, Any]) -> str:
    normalized = _normalize_text(query)
    tool_call = router_plan.get("tool_call") if isinstance(router_plan.get("tool_call"), dict) else {}
    parameters = tool_call.get("parameters") if isinstance(tool_call.get("parameters"), dict) else {}
    formula_ids = parameters.get("formula_ids")
    if isinstance(formula_ids, list) and formula_ids:
        return "formula"
    if router_plan.get("needs_medical_tool") or extracted_tool_payload.get("measurements"):
        return "threshold"
    if any(term in normalized for term in ("la gi", "khai niem", "dinh nghia")):
        return "definition"
    return "medical_qa"


def _reason(
    *,
    query_type: str,
    filters: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    tool_result: dict[str, Any] | None,
) -> str:
    if _matched_threshold_items(tool_result):
        return "tool_result_informed_retrieval"
    if filters.get("disease_name"):
        return f"high_confidence_disease_filter:{filters['disease_name']}"
    if any(candidates.values()):
        return f"soft_hint_retrieval:{query_type}"
    return "plain_hybrid_retrieval"


def _confidence(name: str | None, candidates: list[dict[str, Any]]) -> float | None:
    if not name:
        return None
    for item in candidates:
        if item["name"] == name:
            return item["confidence"]
    return None


def _alias_confidence(alias: str, *, kind: str) -> float:
    token_count = len(alias.split())
    if kind == "biomarker":
        return 0.9 if len(alias) <= 5 else 0.86
    if kind == "section":
        return min(0.88, 0.68 + token_count * 0.06)
    if alias in {"viem cau than", "cau than", "benh cau than"}:
        return 0.62
    if alias in {"lupus", "ara 1997", "class iv"}:
        return 0.9
    if alias in {"acr", "albumin nieu"}:
        return 0.78
    if alias in {"ckd", "aki", "fena", "rifle"}:
        return 0.9
    return min(0.95, 0.68 + token_count * 0.06 + min(len(alias), 18) * 0.006)


def _explicit_filter(value: Any) -> str | None:
    text = _clean_text(value)
    return text or None


def _explicit_section_filter(value: Any) -> str | None:
    text = _clean_text(value)
    return text if text in ALLOWED_SECTION_TYPES else None


def _explicit_source_filter(value: Any) -> str | None:
    text = _clean_text(value)
    return text if text in ALLOWED_SOURCE_TYPES else None


def _canonical_disease_name(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    normalized = _normalize_text(text)
    normalized_key = normalized.replace(" ", "_")
    if text in ALLOWED_RETRIEVAL_DISEASE_NAMES:
        return text
    if normalized_key in ALLOWED_RETRIEVAL_DISEASE_NAMES:
        return normalized_key
    if text in DISEASE_NAME_ALIASES:
        return DISEASE_NAME_ALIASES[text]
    if normalized in DISEASE_NAME_ALIASES:
        return DISEASE_NAME_ALIASES[normalized]
    if normalized_key in DISEASE_NAME_ALIASES:
        return DISEASE_NAME_ALIASES[normalized_key]
    return None


def _contains_term(normalized_text: str, normalized_term: str) -> bool:
    if " " in normalized_term or len(normalized_term) > 4:
        return normalized_term in normalized_text
    return re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", normalized_text) is not None


def _normalize_text(text: Any) -> str:
    raw = str(text or "")
    normalized = unicodedata.normalize("NFKD", raw.replace("đ", "d").replace("Đ", "D"))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return " ".join(ascii_text.split())


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
