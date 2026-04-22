from __future__ import annotations

"""
LangGraph flow tối thiểu cho chatbot VitalAI.

Graph này tách rõ các bước:
- chuẩn bị input
- routing nhẹ
- retrieval khi cần
- build prompt bằng template
- gọi LLM
- cleanup response trước khi trả cho user
"""

import re
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from src.LLM.prompts.tool_router_prompt import MEDICAL_TOOL_ROUTER_PROMPT
from src.LLM.prompts.templates import DIRECT_ANSWER_PROMPT, RAG_ANSWER_PROMPT, STRUCTURED_FINAL_ANSWER_PROMPT
from src.LLM.retrieval.vector_search import NeonVectorSearcher
from src.LLM.tools import (
    MedicalToolsClient,
    build_structured_answer,
    build_structured_context,
    load_medical_tools_contract,
    normalize_router_plan,
    parse_router_plan,
)


RouteName = Literal["retrieve", "direct"]


class ChatbotGraphState(TypedDict, total=False):
    """State đi qua các node trong chatbot graph."""

    query: str
    top_k: int
    disease_name: str | None
    section_type: str | None
    source_type: str | None
    biomarker: str | None
    route: RouteName
    retrieval: dict[str, Any] | None
    evidence_items: list[dict[str, Any]]
    evidence_context: str
    prompt_messages: list[BaseMessage]
    raw_answer: str
    final_answer: str
    safe_structured_answer: str
    used_structured_answer: bool
    user_sources: list[dict[str, Any]]
    debug_results: list[dict[str, Any]]
    query_understanding: dict[str, Any] | None
    filters: dict[str, Any]
    tool_contract: str
    router_plan: dict[str, Any] | None
    router_error: str | None
    medical_tool_result: dict[str, Any] | None
    structured_context: str


def build_chatbot_graph(
    *,
    searcher: NeonVectorSearcher,
    llm: Any,
    max_evidence_items: int = 5,
    medical_tools_base_url: str | None = None,
    tool_contract_path: str | None = None,
) -> Any:
    """Build và compile LangGraph cho QA flow."""

    medical_tools_client = MedicalToolsClient(medical_tools_base_url) if medical_tools_base_url else None

    async def prepare_input(state: ChatbotGraphState) -> ChatbotGraphState:
        query = " ".join(state.get("query", "").split())
        return {
            **state,
            "query": query,
            "top_k": int(state.get("top_k") or 5),
            "router_plan": None,
            "medical_tool_result": None,
            "structured_context": "Không có kết quả phân tích chỉ số.",
            "safe_structured_answer": "",
            "filters": {
                "disease_name": state.get("disease_name"),
                "section_type": state.get("section_type"),
                "source_type": state.get("source_type"),
                "biomarker": state.get("biomarker"),
            },
        }

    def route_input(state: ChatbotGraphState) -> RouteName:
        query = state.get("query", "").lower()
        if _is_direct_query(query):
            return "direct"
        return "retrieve"

    async def route_with_medical_tools(state: ChatbotGraphState) -> ChatbotGraphState:
        heuristic_plan = _build_heuristic_router_plan(state["query"])
        if heuristic_plan:
            return {
                **state,
                "tool_contract": "",
                "router_plan": normalize_router_plan(heuristic_plan, state["query"]),
                "router_error": None,
            }

        try:
            tool_contract = load_medical_tools_contract(tool_contract_path)
            prompt = MEDICAL_TOOL_ROUTER_PROMPT.invoke(
                {
                    "tool_contract": tool_contract,
                    "query": state["query"],
                }
            )
            response = await llm.ainvoke(
                prompt.messages,
                config={"tags": ["internal_router"], "metadata": {"internal": True}},
            )
            raw_plan = parse_router_plan(str(response.content))
            router_plan = normalize_router_plan(raw_plan, state["query"])
            return {**state, "tool_contract": tool_contract, "router_plan": router_plan, "router_error": None}
        except Exception as exc:  # Router failure should degrade to normal RAG.
            fallback_plan = normalize_router_plan(
                {
                    "needs_medical_tool": False,
                    "tool_call": None,
                    "rag_plan": {"should_retrieve": True, "query": state["query"], "filters": {}},
                    "missing_inputs": [],
                    "reason": "router_failed_fallback_to_rag",
                },
                state["query"],
            )
            return {**state, "router_plan": fallback_plan, "router_error": str(exc)}

    async def call_medical_tools(state: ChatbotGraphState) -> ChatbotGraphState:
        router_plan = state.get("router_plan") or {}
        tool_call = router_plan.get("tool_call")
        if not router_plan.get("needs_medical_tool") or not tool_call:
            return {**state, "structured_context": build_structured_context(None, query=state.get("query"))}

        if medical_tools_client is None:
            result = {
                "tool_status": "unavailable",
                "error": "MEDICAL_TOOLS_BASE_URL chưa được cấu hình trong AI service.",
            }
        else:
            result = await medical_tools_client.evaluate(
                parameters=tool_call.get("parameters") or {},
                endpoint=tool_call.get("endpoint") or "/mcp/medical-tools/evaluate",
            )
        return {
            **state,
            "medical_tool_result": result,
            "structured_context": build_structured_context(result, query=state.get("query")),
            "safe_structured_answer": build_structured_answer(result, query=state.get("query")) or "",
        }

    async def retrieve_context(state: ChatbotGraphState) -> ChatbotGraphState:
        router_plan = state.get("router_plan") or {}
        rag_plan = router_plan.get("rag_plan") or {}
        filters = rag_plan.get("filters") or {}
        retrieval_query = rag_plan.get("query") or state["query"]
        disease_name = filters.get("disease_name") or state.get("disease_name")
        section_type = filters.get("section_type") or state.get("section_type")
        source_type = filters.get("source_type") or state.get("source_type")
        biomarker = filters.get("biomarker") or state.get("biomarker")

        tool_retrieval = _build_tool_informed_retrieval(state)
        if tool_retrieval:
            retrieval_query = tool_retrieval["query"]
            disease_name = tool_retrieval.get("disease_name")
            section_type = tool_retrieval.get("section_type")
            source_type = tool_retrieval.get("source_type")
            biomarker = tool_retrieval.get("biomarker")

        retrieval = await searcher.search(
            query=retrieval_query,
            top_k=state["top_k"],
            disease_name=disease_name,
            section_type=section_type,
            source_type=source_type,
            biomarker=biomarker,
        )
        evidence_items = retrieval["results"][:max_evidence_items]
        if state.get("medical_tool_result"):
            evidence_items = _filter_tool_evidence_items(
                evidence_items,
                state.get("medical_tool_result"),
                limit=max_evidence_items,
            )
            if not evidence_items:
                evidence_items = _tool_source_evidence_items(
                    state.get("medical_tool_result"),
                    limit=max_evidence_items,
                )
        return {
            **state,
            "route": "retrieve",
            "retrieval": retrieval,
            "evidence_items": evidence_items,
            "evidence_context": _format_evidence_for_prompt(evidence_items),
            "debug_results": retrieval["results"],
            "query_understanding": retrieval.get("query_understanding"),
            "filters": retrieval.get("filters", state.get("filters", {})),
        }

    async def build_prompt(state: ChatbotGraphState) -> ChatbotGraphState:
        if state.get("retrieval") is None:
            prompt = DIRECT_ANSWER_PROMPT.invoke({"query": state["query"]})
            route: RouteName = "direct"
        elif state.get("medical_tool_result"):
            evidence_context = state.get("evidence_context") or "Không tìm thấy evidence phù hợp."
            safe_structured_answer = state.get("safe_structured_answer") or build_structured_answer(
                state.get("medical_tool_result"),
                query=state.get("query"),
            ) or state.get("structured_context") or "Không có kết quả phân tích chỉ số."
            prompt = STRUCTURED_FINAL_ANSWER_PROMPT.invoke(
                {
                    "query": state["query"],
                    "safe_structured_answer": safe_structured_answer,
                    "evidence_context": evidence_context,
                }
            )
            route = "retrieve"
        else:
            evidence_context = state.get("evidence_context") or "Không tìm thấy evidence phù hợp."
            prompt = RAG_ANSWER_PROMPT.invoke(
                {
                    "query": state["query"],
                    "evidence_context": evidence_context,
                    "structured_context": state.get("structured_context") or "Không có kết quả phân tích chỉ số.",
                }
            )
            route = "retrieve"
        return {**state, "route": route, "prompt_messages": prompt.messages}

    def route_after_retrieval(state: ChatbotGraphState) -> Literal["build_prompt", "generate_response"]:
        return "build_prompt"

    async def generate_response(state: ChatbotGraphState) -> ChatbotGraphState:
        has_structured_result = bool(state.get("medical_tool_result"))
        structured_result = state.get("medical_tool_result") or {}
        structured_result_type = structured_result.get("result_type")
        safe_structured_answer = state.get("safe_structured_answer") or ""
        if structured_result_type in {"structured_knowledge_query", "structured_graph_query"} and safe_structured_answer:
            return {
                **state,
                "raw_answer": safe_structured_answer,
                "used_structured_answer": True,
            }
        if state.get("route") == "retrieve" and not state.get("evidence_items") and not has_structured_result:
            raw_answer = (
                "Mình chưa tìm thấy ngữ cảnh phù hợp trong kho tài liệu hiện tại, "
                "nên chưa thể trả lời chắc chắn cho câu hỏi này."
            )
            used_structured_answer = False
        else:
            try:
                response = await llm.ainvoke(
                    state["prompt_messages"],
                    config={"tags": ["final_answer"], "metadata": {"internal": False}},
                )
                raw_answer = str(response.content).strip()
            except Exception:
                raw_answer = state.get("safe_structured_answer") or build_structured_answer(
                    state.get("medical_tool_result"),
                    query=state.get("query"),
                ) or (
                    "Mình đã xử lý được dữ liệu đầu vào, nhưng chưa thể sinh phần diễn giải cuối. "
                    "Vui lòng thử lại sau."
                )
            used_structured_answer = has_structured_result
        return {**state, "raw_answer": raw_answer, "used_structured_answer": used_structured_answer}

    async def cleanup_response(state: ChatbotGraphState) -> ChatbotGraphState:
        final_answer = cleanup_user_answer(state.get("raw_answer", ""))
        return {
            **state,
            "final_answer": final_answer,
            "user_sources": [] if state.get("used_structured_answer") else _build_user_sources(state.get("evidence_items", [])),
        }

    graph = StateGraph(ChatbotGraphState)
    graph.add_node("prepare_input", prepare_input)
    graph.add_node("route_with_medical_tools", route_with_medical_tools)
    graph.add_node("call_medical_tools", call_medical_tools)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("generate_response", generate_response)
    graph.add_node("cleanup_response", cleanup_response)

    graph.set_entry_point("prepare_input")
    graph.add_conditional_edges(
        "prepare_input",
        route_input,
        {
            "retrieve": "route_with_medical_tools",
            "direct": "build_prompt",
        },
    )
    graph.add_edge("route_with_medical_tools", "call_medical_tools")
    graph.add_edge("call_medical_tools", "retrieve_context")
    graph.add_conditional_edges(
        "retrieve_context",
        route_after_retrieval,
        {
            "build_prompt": "build_prompt",
            "generate_response": "generate_response",
        },
    )
    graph.add_edge("build_prompt", "generate_response")
    graph.add_edge("generate_response", "cleanup_response")
    graph.add_edge("cleanup_response", END)
    return graph.compile()


def cleanup_user_answer(answer: str) -> str:
    """Loại metadata nội bộ khỏi answer cuối cùng trước khi đưa ra UI/user."""

    cleaned = answer.strip()
    patterns = [
        r"\[\s*nguồn\s*\d+\s*\]",
        r"\(\s*nguồn\s*\d+\s*\)",
        r"\btheo\s+nguồn\s*\d+\b[:,]?",
        r"\bnguồn\s*\d+\b[:,]?",
        r"\btài\s+liệu\s+tham\s+khảo\s*\d*\b[:,]?",
        r"</?context_item>",
        r"\bcitation\s*[:=]\s*[^,\]\s]+",
        r"\[source_id=[^\]]+\]",
        r"\[document_id=[^\]]+\]",
        r"\bsource_id\s*=\s*[^,\]\s]+",
        r"\bdocument_id\s*=\s*[^,\]\s]+",
        r"\btr\.\s*\d+\b",
        r"\btrang\s+\d+\b",
        r"\bpage\s*(?:number|index)?\s*[:=]?\s*\d+\b",
        r"\bscore\s*[:=]\s*[0-9.]+\b",
        r"\bsim\s*[:=]\s*[0-9.]+\b",
        r"\bfts\s*[:=]\s*[0-9.]+\b",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btheo\s*[,.:;]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\s*,\s*\]", "", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _build_heuristic_router_plan(query: str) -> dict[str, Any] | None:
    """Route obvious numeric/formula questions without an LLM router call.

    This prevents internal router JSON from being streamed to UI for common
    formula cases and reduces latency/token usage.
    """

    normalized = _normalize_query_text(query)
    if _looks_like_structured_query(normalized):
        source_type = "structured_graph" if _looks_like_graph_query(normalized) else "structured_table"
        return _tool_plan_structured(
            query=query,
            endpoint="/mcp/medical-tools/structured-knowledge-query",
            parameters={"query": query, "top_k": 5},
            rag_query=query,
            source_type=source_type,
            reason="heuristic_structured_first",
        )

    formula_ids = _detect_formula_ids(normalized, query)
    has_threshold_values = bool(
        re.search(r"\d", query)
        and any(
            keyword in normalized
            for keyword in (
                "acr",
                "gfr",
                "creatinine",
                "creatinin",
                "hba1c",
                "ldl",
                "kali",
                "potassium",
                "cholesterol",
                "huyet ap",
            )
        )
    )

    if formula_ids:
        disease_name = "acute_kidney_injury" if "fena_formula" in formula_ids else "benh_than_man"
        if any(keyword in normalized for keyword in ("acr", "gfr", "benh than man", "ckd")):
            disease_name = "benh_than_man"
        elif "fena_formula" in formula_ids:
            disease_name = "acute_kidney_injury"

        rag_query = _formula_rag_query(query=query, formula_ids=formula_ids, has_threshold_values=has_threshold_values)
        return _tool_plan(
            query=query,
            disease_name=disease_name,
            formula_ids=formula_ids,
            rag_query=rag_query,
            section_type="general",
            biomarker=None,
            reason="heuristic_formula_and_threshold_values" if has_threshold_values else "heuristic_formula_values",
        )

    if has_threshold_values:
        return _tool_plan(
            query=query,
            disease_name="benh_than_man" if any(keyword in normalized for keyword in ("acr", "gfr")) else None,
            formula_ids=[],
            rag_query=query,
            section_type="classification",
            biomarker=None,
            reason="heuristic_threshold_values",
        )

    # Structured-first default for knowledge questions without explicit formula inputs.
    return _tool_plan_structured(
        query=query,
        endpoint="/mcp/medical-tools/structured-knowledge-query",
        parameters={"query": query, "top_k": 5},
        rag_query=query,
        source_type=None,
        reason="default_structured_first",
    )


def _tool_plan_structured(
    *,
    query: str,
    endpoint: str,
    parameters: dict[str, Any],
    rag_query: str,
    source_type: str | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "needs_medical_tool": True,
        "tool_call": {
            "tool_name": "medical_tools.evaluate",
            "method": "POST",
            "endpoint": endpoint,
            "parameters": parameters,
        },
        "rag_plan": {
            "should_retrieve": True,
            "query": rag_query,
            "filters": {
                "disease_name": None,
                "section_type": None,
                "source_type": source_type,
                "biomarker": None,
            },
        },
        "missing_inputs": [],
        "reason": reason,
    }


def _looks_like_structured_query(normalized: str) -> bool:
    keywords = (
        "bang",
        "du lieu bang",
        "so do",
        "luu do",
        "decision tree",
        "nhanh",
        "graph",
        "metadata",
        "rifle",
        "risk",
        "injury",
        "failure",
        "loss",
        "end stage",
    )
    return any(keyword in normalized for keyword in keywords)


def _looks_like_graph_query(normalized: str) -> bool:
    keywords = ("so do", "luu do", "decision tree", "graph", "nhanh", "cay")
    return any(keyword in normalized for keyword in keywords)


def _detect_formula_ids(normalized: str, original_query: str) -> list[str]:
    formula_ids: list[str] = []
    if "fena" in normalized or _has_fena_input_set(normalized):
        formula_ids.append("fena_formula")

    has_creatinine = any(keyword in normalized for keyword in ("creatinine mau", "creatinin mau", "creatinine", "creatinin"))
    has_age = bool(re.search(r"\b\d{1,3}\s*tuoi\b", normalized)) or " tuoi" in normalized or "age" in normalized
    has_sex = any(keyword in normalized for keyword in (" nam ", "nu ", "gioi tinh", "male", "female")) or normalized.startswith("nam ") or normalized.startswith("nu ")
    has_race = "race" in normalized or "chung toc" in normalized
    has_weight = any(keyword in normalized for keyword in ("nang", "can nang", "kg", "weight"))
    has_height = any(keyword in normalized for keyword in ("cao", "chieu cao", "cm", "height"))

    if any(keyword in normalized for keyword in ("mdrd", "egfr", "e gfr", "tinh gfr")) or (has_creatinine and has_age and has_sex and has_race):
        formula_ids.append("mdrd_gfr")
    if any(keyword in normalized for keyword in ("cockcroft", "gault")) or (has_creatinine and has_age and has_sex and has_weight):
        formula_ids.append("cockcroft_gault")
    if any(keyword in normalized for keyword in ("bsa", "dien tich da")) or (has_height and has_weight):
        formula_ids.append("body_surface_area")

    return _unique_non_empty(formula_ids)


def _formula_rag_query(*, query: str, formula_ids: list[str], has_threshold_values: bool) -> str:
    labels = {
        "fena_formula": "FENa trong chẩn đoán suy thận cấp",
        "mdrd_gfr": "MDRD eGFR và phân loại bệnh thận mạn",
        "cockcroft_gault": "Cockcroft-Gault creatinine clearance",
        "body_surface_area": "diện tích da cơ thể BSA",
    }
    parts = [query]
    parts.extend(labels[item] for item in formula_ids if item in labels)
    if has_threshold_values:
        parts.append("đánh giá ngưỡng ACR GFR kali huyết áp cholesterol nếu có")
    return "\n".join(parts)


def _build_tool_informed_retrieval(state: ChatbotGraphState) -> dict[str, Any] | None:
    """Create a focused RAG query after formulas/thresholds are available."""

    result = state.get("medical_tool_result") or {}
    if not result or result.get("tool_status"):
        return None

    matched_items = [item for item in result.get("threshold_matches", []) if isinstance(item, dict) and item.get("matched")]
    formula_items = [item for item in result.get("formula_results", []) if isinstance(item, dict)]
    derived_items = [item for item in result.get("derived_measurements", []) if isinstance(item, dict)]
    if not matched_items and not formula_items and not derived_items:
        return None

    disease_names = _unique_non_empty(
        (item.get("threshold") or {}).get("disease_name") for item in matched_items
    )
    biomarkers = _unique_non_empty(
        [item.get("biomarker") for item in matched_items]
        + [item.get("name") for item in derived_items]
    )
    labels = _unique_non_empty((item.get("threshold") or {}).get("label") for item in matched_items)
    formulas = _unique_non_empty(item.get("formula_name") or item.get("formula_id") for item in formula_items)
    conditions = _threshold_condition_phrases(matched_items)
    source_texts = _unique_non_empty((item.get("source") or {}).get("source_text") for item in matched_items)

    query_parts = [state.get("query", "")]
    if formulas:
        query_parts.append("Công thức: " + ", ".join(formulas[:3]))
    if biomarkers:
        query_parts.append("Chỉ số: " + ", ".join(biomarkers[:5]))
    if conditions:
        query_parts.append("Ngưỡng/phân loại: " + "; ".join(conditions[:5]))
    if labels:
        query_parts.append("Ý nghĩa lâm sàng: " + ", ".join(labels[:5]))
    if source_texts:
        query_parts.append("Đoạn liên quan: " + " ".join(source_texts[:2]))

    return {
        "query": "\n".join(part for part in query_parts if part),
        "disease_name": disease_names[0] if len(disease_names) == 1 else None,
        "section_type": None,
        "source_type": None,
        "biomarker": _safe_tool_rag_biomarker(biomarkers),
    }


def _safe_tool_rag_biomarker(biomarkers: list[str]) -> str | None:
    """Avoid over-filtering RAG when source metadata uses related biomarkers."""

    if len(biomarkers) != 1:
        return None
    biomarker = biomarkers[0]
    if biomarker == "FENa":
        # FENa explanation chunks are often tagged as sodium/formula, not FENa.
        return None
    return biomarker


def _threshold_condition_phrases(items: list[dict[str, Any]]) -> list[str]:
    phrases: list[str] = []
    for item in items:
        threshold = item.get("threshold") or {}
        biomarker = str(item.get("biomarker") or "").strip()
        label = str(threshold.get("label") or "").strip()
        op = threshold.get("op")
        unit = str(threshold.get("unit") or item.get("comparison_unit") or "").strip()
        if op == "between":
            min_value = threshold.get("value_min")
            max_value = threshold.get("value_max")
            condition = f"{min_value}-{max_value} {unit}".strip()
        else:
            condition = f"{op} {threshold.get('value')} {unit}".strip()
        phrase = " ".join(part for part in (biomarker, condition, label) if part)
        if phrase:
            phrases.append(phrase)
    return _unique_non_empty(phrases)


def _filter_tool_evidence_items(
    items: list[dict[str, Any]],
    tool_result: dict[str, Any] | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not tool_result:
        return items[:limit]

    keywords = _tool_evidence_keywords(tool_result)
    requires_fena = _tool_result_has_biomarker(tool_result, "FENa")
    filtered: list[dict[str, Any]] = []
    for item in items:
        text = _normalize_query_text(str(item.get("preview") or item.get("content") or ""))
        if _looks_like_noisy_chunk(text):
            continue
        if requires_fena and "fena" not in text:
            continue
        if keywords and not any(keyword in text for keyword in keywords):
            continue
        filtered.append(item)
    return filtered[:limit]


def _tool_source_evidence_items(tool_result: dict[str, Any] | None, *, limit: int) -> list[dict[str, Any]]:
    if not tool_result:
        return []

    source_texts = _unique_non_empty(
        (item.get("source") or {}).get("source_text")
        for item in tool_result.get("threshold_matches", [])
        if isinstance(item, dict) and item.get("matched")
    )
    evidence_items: list[dict[str, Any]] = []
    for index, text in enumerate(source_texts[:limit], start=1):
        evidence_items.append(
            {
                "document_id": f"tool_source::{index}",
                "source_type": "tool_source",
                "source_id": f"tool_source::{index}",
                "preview": text,
                "disease_name": None,
                "section_type": None,
                "doc_type": "tool_threshold_context",
                "biomarker": None,
            }
        )
    return evidence_items


def _tool_evidence_keywords(tool_result: dict[str, Any]) -> list[str]:
    raw_keywords: list[str] = []
    for item in tool_result.get("threshold_matches", []):
        if not isinstance(item, dict) or not item.get("matched"):
            continue
        threshold = item.get("threshold") or {}
        raw_keywords.extend(
            [
                item.get("biomarker"),
                threshold.get("label"),
                threshold.get("disease_name"),
                (item.get("source") or {}).get("source_text"),
            ]
        )
    for item in tool_result.get("formula_results", []):
        if isinstance(item, dict):
            raw_keywords.extend([item.get("formula_name"), item.get("formula_id"), item.get("output_name")])

    keywords: list[str] = []
    for value in raw_keywords:
        normalized = _normalize_query_text(str(value or ""))
        for token in re.findall(r"[a-z0-9_]{3,}", normalized):
            if token in {"acute", "kidney", "injury", "benh", "than", "man", "formula", "threshold"}:
                continue
            keywords.append(token)
    return _unique_non_empty(keywords)


def _tool_result_has_biomarker(tool_result: dict[str, Any], biomarker: str) -> bool:
    for key in ("threshold_matches", "classifications"):
        for item in tool_result.get(key, []):
            if isinstance(item, dict) and item.get("biomarker") == biomarker:
                return True
    for item in tool_result.get("derived_measurements", []):
        if isinstance(item, dict) and item.get("name") == biomarker:
            return True
    return False


def _looks_like_noisy_chunk(text: str) -> bool:
    if not text:
        return True
    brace_count = text.count("{") + text.count("}") + text.count("]") + text.count("[")
    return brace_count >= 6 and len(text) < 800


def _unique_non_empty(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _tool_plan(
    *,
    query: str,
    disease_name: str | None,
    formula_ids: list[str],
    rag_query: str,
    section_type: str | None,
    biomarker: str | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "needs_medical_tool": True,
        "tool_call": {
            "tool_name": "medical_tools.evaluate",
            "method": "POST",
            "endpoint": "/mcp/medical-tools/evaluate",
            "parameters": {
                "text": query,
                "measurements": None,
                "disease_name": disease_name,
                "formula_ids": formula_ids,
                "include_debug": False,
            },
        },
        "rag_plan": {
            "should_retrieve": True,
            "query": rag_query,
            "filters": {
                "disease_name": disease_name,
                "section_type": section_type,
                "source_type": "chunk",
                "biomarker": biomarker,
            },
        },
        "missing_inputs": [],
        "reason": reason,
    }


def _looks_like_internal_router_json(answer: str) -> bool:
    normalized = answer.strip().lower()
    return normalized.startswith("{") and "needs_medical_tool" in normalized and "tool_call" in normalized


def _looks_like_internal_leak(answer: str) -> bool:
    normalized = answer.lower()
    blocked_terms = (
        "needs_medical_tool",
        "tool_call",
        "router_plan",
        "medical_tool_result",
        "structured_context",
        "source_id",
        "document_id",
        "threshold_id",
        "mcp",
        "endpoint",
        "context_item",
    )
    return any(term in normalized for term in blocked_terms) or normalized.startswith("```json")


def _has_unsupported_structured_expansion(answer: str, safe_facts: str, evidence_context: str = "") -> bool:
    """Detect common medical expansions that the final LLM may add beyond facts."""

    if not safe_facts:
        return False
    answer_norm = _normalize_query_text(answer)
    allowed_context = _normalize_query_text(f"{safe_facts}\n{evidence_context}")
    risky_terms = (
        "mat nuoc",
        "giam the tich",
        "tuan hoan",
        "tieu duong",
        "huyet ap",
        "met moi",
        "phu",
        "tieu it",
        "dau dau",
        "dieu tri",
        "huong dieu tri",
        "nguyen nhan",
        "proteinuria",
        "ro protein",
        "than ro protein",
        "protein nieu",
    )
    return any(term in answer_norm and term not in allowed_context for term in risky_terms)


def _has_fena_input_set(normalized_query: str) -> bool:
    """Detect the four direct inputs needed for FENa without requiring the word FENa."""

    has_urine_na = any(token in normalized_query for token in ("na nieu", "natri nieu", "urine na", "urine_na", "una"))
    has_plasma_na = any(token in normalized_query for token in ("na mau", "natri mau", "plasma na", "plasma_na", "pna"))
    has_urine_creatinine = any(
        token in normalized_query
        for token in (
            "creatinine nieu",
            "creatinin nieu",
            "urine creatinine",
            "urine_creatinine",
            "ucr",
        )
    )
    has_plasma_creatinine = any(
        token in normalized_query
        for token in (
            "creatinine mau",
            "creatinin mau",
            "plasma creatinine",
            "plasma_creatinine",
            "pcr",
        )
    )
    return has_urine_na and has_plasma_na and has_urine_creatinine and has_plasma_creatinine


def _normalize_query_text(query: str) -> str:
    normalized = query.lower()
    replacements = {
        "đ": "d",
        "á": "a",
        "à": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "ă": "a",
        "ắ": "a",
        "ằ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "ặ": "a",
        "â": "a",
        "ấ": "a",
        "ầ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ậ": "a",
        "é": "e",
        "è": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ê": "e",
        "ế": "e",
        "ề": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "í": "i",
        "ì": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ó": "o",
        "ò": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ô": "o",
        "ố": "o",
        "ồ": "o",
        "ổ": "o",
        "ỗ": "o",
        "ộ": "o",
        "ơ": "o",
        "ớ": "o",
        "ờ": "o",
        "ở": "o",
        "ỡ": "o",
        "ợ": "o",
        "ú": "u",
        "ù": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ư": "u",
        "ứ": "u",
        "ừ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "ý": "y",
        "ỳ": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def _is_direct_query(query: str) -> bool:
    """Routing nhẹ: chỉ bỏ qua RAG cho lời chào/cảm ơn/hỏi khả năng hệ thống."""

    normalized = _normalize_query_text(query)
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = " ".join(normalized.split())

    medical_keywords = (
        "la gi",
        "khai niem",
        "dinh nghia",
        "lupus",
        "than",
        "benh",
        "acr",
        "gfr",
        "creatinine",
        "creatinin",
        "fena",
        "trieu chung",
        "dieu tri",
        "chan doan",
    )
    if any(keyword in normalized for keyword in medical_keywords):
        return False

    direct_exact = {
        "hi",
        "hello",
        "hey",
        "xin chao",
        "chao",
        "chao ban",
        "cam on",
        "thanks",
        "thank you",
        "ban la ai",
        "ban lam duoc gi",
    }
    return normalized in direct_exact


def _format_evidence_for_prompt(evidence_items: list[dict[str, Any]]) -> str:
    """Format evidence cho prompt nhưng không đưa số trang, source label hoặc id nội bộ vào."""

    if not evidence_items:
        return "Không tìm thấy evidence phù hợp."

    blocks: list[str] = []
    for item in evidence_items:
        preview = (item.get("preview") or "").strip()
        if preview:
            blocks.append(f"<context_item>\n{preview}\n</context_item>")
    return "\n\n".join(blocks) if blocks else "Không tìm thấy evidence phù hợp."


def _build_user_sources(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tạo source metadata an toàn cho UI, không chứa page/source_id/document_id."""

    sources: list[dict[str, Any]] = []
    for index, item in enumerate(evidence_items, start=1):
        sources.append(
            {
                "label": f"Tài liệu tham khảo {index}",
                "source_type": item.get("source_type"),
                "section_type": item.get("section_type"),
                "disease_name": item.get("disease_name"),
                "preview": item.get("preview", "")[:240],
            }
        )
    return sources
