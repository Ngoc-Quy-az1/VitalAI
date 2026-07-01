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
import json
import asyncio
import os
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from src.LLM.prompts.tool_router_prompt import MEDICAL_TOOL_ROUTER_PROMPT
from src.LLM.prompts.templates import DIRECT_ANSWER_PROMPT, RAG_ANSWER_PROMPT
from src.LLM.retrieval.query_planner import build_retrieval_plan
from src.LLM.retrieval.vector_search import DISEASE_LABELS, SECTION_LABELS, NeonVectorSearcher
from src.LLM.tools import (
    MedicalToolsClient,
    build_structured_context,
    build_supported_tool_context,
    build_tool_input_payload,
    load_medical_tools_contract,
    normalize_router_plan,
    parse_router_plan,
    tool_payload_has_supported_inputs,
)
from src.LLM.web_search import search_medical_web


RouteName = Literal["retrieve", "direct"]


class ChatbotGraphState(TypedDict, total=False):
    """State đi qua các node trong chatbot graph."""

    query: str
    top_k: int
    disease_name: str | None
    section_type: str | None
    source_type: str | None
    biomarker: str | None
    conversation_id: str | None
    user_id: str | None
    memory_context: str
    enable_web_search: bool | None
    route: RouteName
    retrieval: dict[str, Any] | None
    evidence_items: list[dict[str, Any]]
    evidence_context: str
    prompt_messages: list[BaseMessage]
    raw_answer: str
    final_answer: str
    user_sources: list[dict[str, Any]]
    debug_results: list[dict[str, Any]]
    query_understanding: dict[str, Any] | None
    retrieval_plan: dict[str, Any] | None
    filters: dict[str, Any]
    tool_contract: str
    router_plan: dict[str, Any] | None
    router_error: str | None
    medical_tool_result: dict[str, Any] | None
    structured_context: str
    extracted_tool_payload: dict[str, Any]
    supported_tool_context: str
    web_context: str
    web_results: list[dict[str, Any]]
    evidence_quality: dict[str, Any]
    evidence_grades: list[dict[str, Any]]
    agentic_queries: list[str]
    agentic_retry_count: int
    agentic_retry_query: str | None
    chat_history: list[dict[str, str]]


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
        chat_history = state.get("chat_history") or []
        enriched_query = _enrich_query_with_history(query, chat_history)
        return {
            **state,
            "query": enriched_query,
            "top_k": int(state.get("top_k") or 5),
            "router_plan": None,
            "medical_tool_result": None,
            "structured_context": "Không có kết quả phân tích chỉ số.",
            "web_context": "Không có ngữ cảnh web y tế bổ sung.",
            "web_results": [],
            "evidence_quality": {},
            "evidence_grades": [],
            "agentic_queries": [],
            "agentic_retry_count": int(state.get("agentic_retry_count") or 0),
            "agentic_retry_query": None,
            "memory_context": _trim_text(state.get("memory_context") or "", 1600)
            or "Không có memory ngắn hạn.",
            "chat_history": state.get("chat_history") or [],
            "conversation_id": state.get("conversation_id"),
            "user_id": state.get("user_id"),
            "enable_web_search": state.get("enable_web_search"),
            "retrieval_plan": None,
            "extracted_tool_payload": {"text": query},
            "supported_tool_context": build_supported_tool_context(),
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

    async def extract_tool_payload(state: ChatbotGraphState) -> ChatbotGraphState:
        payload = build_tool_input_payload(state["query"])
        return {
            **state,
            "extracted_tool_payload": payload,
            "supported_tool_context": build_supported_tool_context(),
        }

    async def route_with_medical_tools(state: ChatbotGraphState) -> ChatbotGraphState:
        heuristic_plan = _build_heuristic_router_plan(state["query"], state.get("extracted_tool_payload"))
        if heuristic_plan:
            return {
                **state,
                "tool_contract": "",
                "router_plan": normalize_router_plan(
                    heuristic_plan,
                    state["query"],
                    extracted_payload=state.get("extracted_tool_payload"),
                ),
                "router_error": None,
            }

        try:
            tool_contract = load_medical_tools_contract(tool_contract_path)
            prompt = MEDICAL_TOOL_ROUTER_PROMPT.invoke(
                {
                    "tool_contract": tool_contract,
                    "supported_tool_context": state.get("supported_tool_context") or build_supported_tool_context(),
                    "extracted_tool_payload": json.dumps(
                        state.get("extracted_tool_payload") or {"text": state["query"]},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "query": state["query"],
                }
            )
            response = await llm.ainvoke(
                prompt.messages,
                config={"tags": ["internal_router"], "metadata": {"internal": True}},
            )
            raw_plan = parse_router_plan(str(response.content))
            router_plan = normalize_router_plan(
                raw_plan,
                state["query"],
                extracted_payload=state.get("extracted_tool_payload"),
            )
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
                extracted_payload=state.get("extracted_tool_payload"),
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
        }

    async def understand_retrieval_query(state: ChatbotGraphState) -> ChatbotGraphState:
        plan = build_retrieval_plan(
            query=state["query"],
            initial_filters={
                "disease_name": state.get("disease_name"),
                "section_type": state.get("section_type"),
                "source_type": state.get("source_type"),
                "biomarker": state.get("biomarker"),
            },
            router_plan=state.get("router_plan"),
            extracted_tool_payload=state.get("extracted_tool_payload"),
            medical_tool_result=state.get("medical_tool_result"),
            chat_history=state.get("chat_history"),
            memory_context=state.get("memory_context"),
        )
        return {
            **state,
            "retrieval_plan": plan,
            "agentic_queries": _build_agentic_subqueries(state["query"], plan),
            "filters": plan.get("filters", state.get("filters", {})),
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

        retrieval_plan = state.get("retrieval_plan") or {}
        if retrieval_plan:
            plan_filters = retrieval_plan.get("filters") or {}
            retrieval_query = retrieval_plan.get("query") or retrieval_query
            disease_name = plan_filters.get("disease_name")
            section_type = plan_filters.get("section_type")
            source_type = plan_filters.get("source_type")
            biomarker = plan_filters.get("biomarker")

        tool_retrieval = _build_tool_informed_retrieval(state)
        if tool_retrieval and not retrieval_plan:
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
        retrieval_results = retrieval["results"]
        for subquery in state.get("agentic_queries", []):
            if not subquery or subquery == retrieval_query:
                continue
            try:
                sub_retrieval = await searcher.search(
                    query=subquery,
                    top_k=max(3, min(state["top_k"], max_evidence_items)),
                    disease_name=disease_name,
                    section_type=section_type,
                    source_type=source_type,
                    biomarker=biomarker,
                )
            except Exception:
                continue
            retrieval_results = _merge_agentic_retry_items(
                retrieval_results,
                sub_retrieval.get("results", []),
                limit=max(state["top_k"] * 4, max_evidence_items),
            )

        evidence_items = retrieval_results[:max_evidence_items]
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
        evidence_items, evidence_grades = _grade_and_sort_evidence_items(
            evidence_items,
            state,
            limit=max_evidence_items,
        )
        return {
            **state,
            "route": "retrieve",
            "retrieval": retrieval,
            "evidence_items": evidence_items,
            "evidence_grades": evidence_grades,
            "evidence_context": _format_evidence_for_prompt(evidence_items),
            "debug_results": retrieval_results,
            "query_understanding": {
                "retrieval_planner": retrieval_plan,
                "searcher": retrieval.get("query_understanding"),
            },
            "filters": retrieval.get("filters", state.get("filters", {})),
        }

    async def assess_and_refine_evidence(state: ChatbotGraphState) -> ChatbotGraphState:
        quality = _assess_evidence_quality(state)
        if not _should_agentic_retry(state, quality):
            return {**state, "evidence_quality": quality}

        retry_query = _build_agentic_retry_query(state)
        try:
            retry_retrieval = await searcher.search(
                query=retry_query,
                top_k=max(state["top_k"], max_evidence_items),
                disease_name=None,
                section_type=None,
                source_type="chunk",
                biomarker=None,
            )
            retry_results = retry_retrieval.get("results", [])
        except Exception as exc:
            retry_retrieval = {"results": [], "error": str(exc)}
            retry_results = []

        merged_items = _merge_agentic_retry_items(
            state.get("evidence_items", []),
            retry_results[:max_evidence_items],
            limit=max_evidence_items,
        )
        merged_items, evidence_grades = _grade_and_sort_evidence_items(
            merged_items,
            state,
            limit=max_evidence_items,
        )
        merged_debug_results = _merge_agentic_retry_items(
            state.get("debug_results", []),
            retry_results,
            limit=max(state["top_k"] * 4, max_evidence_items),
        )
        refined_state: ChatbotGraphState = {
            **state,
            "agentic_retry_count": int(state.get("agentic_retry_count") or 0) + 1,
            "agentic_retry_query": retry_query,
            "evidence_items": merged_items,
            "evidence_grades": evidence_grades,
            "evidence_context": _format_evidence_for_prompt(merged_items),
            "debug_results": merged_debug_results,
            "evidence_quality": {
                **quality,
                "retried": True,
                "retry_query": retry_query,
                "retry_result_count": len(retry_results),
            },
        }
        return {
            **refined_state,
            "evidence_quality": {
                **refined_state["evidence_quality"],
                "after_retry": _assess_evidence_quality(refined_state),
            },
        }

    async def retrieve_medical_web_context(state: ChatbotGraphState) -> ChatbotGraphState:
        if not _web_search_enabled(state):
            return {
                **state,
                "web_context": "Không có ngữ cảnh web y tế bổ sung.",
                "web_results": [],
            }

        retrieval_plan = state.get("retrieval_plan") or {}
        query = retrieval_plan.get("query") or state.get("query") or ""
        try:
            results = await asyncio.to_thread(search_medical_web, query, num_results=3)
        except Exception:
            results = []
        result_dicts = [item.to_dict() for item in results]
        return {
            **state,
            "web_results": result_dicts,
            "web_context": _format_web_context(result_dicts),
        }

    async def build_prompt(state: ChatbotGraphState) -> ChatbotGraphState:
        if state.get("retrieval") is None:
            prompt = DIRECT_ANSWER_PROMPT.invoke({"query": state["query"]})
            route: RouteName = "direct"
        else:
            evidence_context = state.get("evidence_context") or "Không tìm thấy evidence phù hợp."
            chat_history_list = state.get("chat_history") or []
            chat_history_str = "\n".join(
                f"{'Người dùng' if msg.get('role') == 'user' else 'Trợ lý'}: {msg.get('content', '')}"
                for msg in chat_history_list
            ) or "Không có lịch sử trò chuyện gần đây."
            prompt = RAG_ANSWER_PROMPT.invoke(
                {
                    "query": state["query"],
                    "chat_history": chat_history_str,
                    "evidence_context": evidence_context,
                    "structured_context": state.get("structured_context") or "Không có kết quả phân tích chỉ số.",
                    "memory_context": state.get("memory_context") or "Không có memory ngắn hạn.",
                    "web_context": state.get("web_context") or "Không có ngữ cảnh web y tế bổ sung.",
                }
            )
            route = "retrieve"
        return {**state, "route": route, "prompt_messages": prompt.messages}

    def route_after_retrieval(state: ChatbotGraphState) -> Literal["build_prompt", "generate_response"]:
        return "build_prompt"

    async def generate_response(state: ChatbotGraphState) -> ChatbotGraphState:
        if (
            state.get("route") == "retrieve"
            and not state.get("evidence_items")
            and not state.get("medical_tool_result")
            and not state.get("web_results")
        ):
            raw_answer = (
                "Mình chưa tìm thấy ngữ cảnh phù hợp trong kho tài liệu hiện tại, "
                "nên chưa thể trả lời chắc chắn cho câu hỏi này."
            )
        else:
            try:
                response = await llm.ainvoke(
                    state["prompt_messages"],
                    config={"tags": ["final_answer"], "metadata": {"internal": False}},
                )
                raw_answer = str(response.content).strip()
            except Exception:
                raw_answer = (
                    "Mình đã xử lý được dữ liệu đầu vào, nhưng chưa thể sinh phần diễn giải cuối. "
                    "Vui lòng thử lại sau."
                )
        return {**state, "raw_answer": raw_answer}

    async def cleanup_response(state: ChatbotGraphState) -> ChatbotGraphState:
        final_answer = cleanup_user_answer(state.get("raw_answer", ""))
        return {
            **state,
            "final_answer": final_answer,
            "user_sources": _build_user_sources(
                state.get("evidence_items", []),
                state.get("web_results", []),
            ),
        }

    graph = StateGraph(ChatbotGraphState)
    graph.add_node("prepare_input", prepare_input)
    graph.add_node("extract_tool_payload", extract_tool_payload)
    graph.add_node("route_with_medical_tools", route_with_medical_tools)
    graph.add_node("call_medical_tools", call_medical_tools)
    graph.add_node("understand_retrieval_query", understand_retrieval_query)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("assess_and_refine_evidence", assess_and_refine_evidence)
    graph.add_node("retrieve_medical_web_context", retrieve_medical_web_context)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("generate_response", generate_response)
    graph.add_node("cleanup_response", cleanup_response)

    graph.set_entry_point("prepare_input")
    graph.add_edge("prepare_input", "extract_tool_payload")
    graph.add_conditional_edges(
        "extract_tool_payload",
        route_input,
        {
            "retrieve": "route_with_medical_tools",
            "direct": "build_prompt",
        },
    )
    graph.add_edge("route_with_medical_tools", "call_medical_tools")
    graph.add_edge("call_medical_tools", "understand_retrieval_query")
    graph.add_edge("understand_retrieval_query", "retrieve_context")
    graph.add_edge("retrieve_context", "assess_and_refine_evidence")
    graph.add_edge("assess_and_refine_evidence", "retrieve_medical_web_context")
    graph.add_conditional_edges(
        "retrieve_medical_web_context",
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


def _build_heuristic_router_plan(query: str, extracted_tool_payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
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
    has_threshold_values = tool_payload_has_supported_inputs(extracted_tool_payload)

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
            extracted_tool_payload=extracted_tool_payload,
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
            extracted_tool_payload=extracted_tool_payload,
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
    asks_to_calculate = any(
        keyword in normalized
        for keyword in (
            "tinh",
            "cong thuc",
            "formula",
            "uoc tinh",
            "estimate",
            "calculate",
        )
    )
    mentions_interpretation = any(
        keyword in normalized
        for keyword in (
            "danh gia",
            "phan loai",
            "xep loai",
            "y nghia",
            "giai thich",
        )
    )

    explicit_fena = "fena" in normalized or "fractional excretion of sodium" in normalized
    if explicit_fena or (_has_fena_input_set(normalized) and asks_to_calculate):
        formula_ids.append("fena_formula")

    has_creatinine = any(keyword in normalized for keyword in ("creatinine mau", "creatinin mau", "creatinine", "creatinin"))
    has_age = bool(re.search(r"\b\d{1,3}\s*tuoi\b", normalized)) or " tuoi" in normalized or "age" in normalized
    has_sex = any(keyword in normalized for keyword in (" nam ", "nu ", "gioi tinh", "male", "female")) or normalized.startswith("nam ") or normalized.startswith("nu ")
    has_race = "race" in normalized or "chung toc" in normalized
    has_weight = any(keyword in normalized for keyword in ("nang", "can nang", "kg", "weight"))
    has_height = any(keyword in normalized for keyword in ("cao", "chieu cao", "cm", "height"))

    explicit_mdrd = "mdrd" in normalized
    explicit_egfr = any(keyword in normalized for keyword in ("egfr", "e gfr", "tinh gfr", "uoc tinh gfr"))
    explicit_cockcroft = any(
        keyword in normalized
        for keyword in (
            "cockcroft",
            "gault",
            "do thanh thai creatinine",
            "creatinine clearance",
        )
    )
    explicit_bsa = any(keyword in normalized for keyword in ("bsa", "dien tich da"))
    asks_for_renal_assessment = any(
        keyword in normalized
        for keyword in (
            "danh gia chuc nang than",
            "chuc nang than",
            "danh gia than",
            "co bat thuong khong",
        )
    )

    if explicit_mdrd:
        formula_ids.append("mdrd_gfr")
    elif explicit_egfr or (
        has_creatinine
        and has_age
        and has_sex
        and (
            asks_for_renal_assessment
            or (
                asks_to_calculate
                and not mentions_interpretation
                and has_race
                and any(keyword in normalized for keyword in ("gfr", "egfr"))
            )
        )
    ):
        formula_ids.append("ckd_epi_2021_creatinine")
    if explicit_cockcroft:
        formula_ids.append("cockcroft_gault")
    if explicit_bsa:
        formula_ids.append("body_surface_area")

    return _unique_non_empty(formula_ids)


def _formula_rag_query(*, query: str, formula_ids: list[str], has_threshold_values: bool) -> str:
    labels = {
        "fena_formula": "FENa trong chẩn đoán suy thận cấp",
        "ckd_epi_2021_creatinine": "CKD-EPI 2021 eGFR và phân loại bệnh thận mạn",
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
        text = _normalize_query_text(str(item.get("content") or item.get("preview") or ""))
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
    formula_texts = _unique_non_empty(
        item.get("source_text")
        for item in tool_result.get("formula_results", [])
        if isinstance(item, dict)
    )
    source_texts = _unique_non_empty([*source_texts, *formula_texts])
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
    extracted_tool_payload: dict[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    extracted_payload = dict(extracted_tool_payload or {})
    parameters: dict[str, Any] = {
        "text": extracted_payload.get("text") or query,
        "formula_ids": formula_ids,
        "include_debug": False,
    }
    measurements = extracted_payload.get("measurements")
    if isinstance(measurements, dict) and measurements:
        parameters["measurements"] = measurements
    chosen_disease_name = disease_name or extracted_payload.get("disease_name")
    if chosen_disease_name:
        parameters["disease_name"] = chosen_disease_name

    return {
        "needs_medical_tool": True,
        "tool_call": {
            "tool_name": "medical_tools.evaluate",
            "method": "POST",
            "endpoint": "/mcp/medical-tools/evaluate",
            "parameters": parameters,
        },
        "rag_plan": {
            "should_retrieve": True,
            "query": rag_query,
            "filters": {
                "disease_name": chosen_disease_name,
                "section_type": section_type,
                "source_type": "chunk",
                "biomarker": biomarker,
            },
        },
        "missing_inputs": [],
        "reason": reason,
    }


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


def _enrich_query_with_history(query: str, chat_history: list[dict[str, str]]) -> str:
    normalized_query = _normalize_query_text(query)
    has_disease = False
    from src.LLM.retrieval.vector_search import DISEASE_HINTS
    for aliases in DISEASE_HINTS.values():
        if any(alias in normalized_query for alias in aliases):
            has_disease = True
            break
    
    if not has_disease and chat_history:
        for msg in reversed(chat_history[-3:]):
            content_norm = _normalize_query_text(msg.get("content", ""))
            for canonical, aliases in DISEASE_HINTS.items():
                for alias in aliases:
                    if alias in content_norm:
                        return f"{query} ({alias})"
    return query


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


def _assess_evidence_quality(state: ChatbotGraphState) -> dict[str, Any]:
    """Heuristic evidence judge used before spending tokens on final answer."""

    evidence_items = state.get("evidence_items", [])
    combined_context = " ".join(
        str(item.get("content") or item.get("preview") or "")
        for item in evidence_items
        if isinstance(item, dict)
    )
    normalized_context = _normalize_query_text(combined_context)
    normalized_query = _normalize_query_text(state.get("query", ""))
    query_tokens = _important_query_tokens(normalized_query)
    context_tokens = set(_important_query_tokens(normalized_context, max_tokens=300))
    overlap = len(set(query_tokens) & context_tokens)
    denominator = max(1, min(len(query_tokens), 10))
    token_coverage = round(overlap / denominator, 3)

    retrieval_plan = state.get("retrieval_plan") or {}
    soft_hints = retrieval_plan.get("soft_hints") or {}
    required_terms = [
        *[str(term) for term in soft_hints.get("terms", [])[:8]],
        *[str(term) for term in soft_hints.get("disease_names", [])[:4]],
        *[str(term) for term in soft_hints.get("section_types", [])[:4]],
        *[str(term) for term in soft_hints.get("biomarkers", [])[:4]],
    ]
    term_hits = [
        term
        for term in required_terms
        if _normalize_query_text(term) and _normalize_query_text(term) in normalized_context
    ]
    top_score = max(
        (
            float(item.get("fusion_score") or item.get("keyword_score") or item.get("similarity") or 0.0)
            for item in evidence_items
            if isinstance(item, dict)
        ),
        default=0.0,
    )
    sufficient = bool(evidence_items) and (
        bool(state.get("medical_tool_result"))
        or token_coverage >= 0.34
        or len(term_hits) >= 2
        or top_score >= 0.12
    )
    return {
        "sufficient": sufficient,
        "evidence_count": len(evidence_items),
        "token_coverage": token_coverage,
        "query_token_hits": overlap,
        "term_hits": term_hits[:8],
        "top_score": round(top_score, 6),
    }


def _build_agentic_subqueries(query: str, retrieval_plan: dict[str, Any]) -> list[str]:
    """Deterministically decompose a broad medical question into focused retrievals."""

    soft_hints = retrieval_plan.get("soft_hints") or {}
    disease_names = [DISEASE_LABELS.get(str(item), str(item)) for item in soft_hints.get("disease_names", [])][:3]
    section_types = [SECTION_LABELS.get(str(item), str(item)) for item in soft_hints.get("section_types", [])][:3]
    biomarkers = [str(item) for item in soft_hints.get("biomarkers", []) if item][:4]
    terms = [str(item) for item in soft_hints.get("terms", []) if item][:8]
    base = " ".join(str(query or "").split())

    subqueries: list[str] = []
    if disease_names or section_types:
        subqueries.append(
            "\n".join(
                part
                for part in (
                    base,
                    "Bệnh trọng tâm: " + ", ".join(disease_names) if disease_names else "",
                    "Mục cần tìm: " + ", ".join(section_types) if section_types else "",
                )
                if part
            )
        )
    if biomarkers or terms:
        subqueries.append(
            "\n".join(
                part
                for part in (
                    base,
                    "Chỉ số/từ khóa: " + ", ".join([*biomarkers, *terms][:10]),
                    "Ưu tiên đoạn có tiêu chuẩn, ngưỡng, biểu hiện hoặc định nghĩa trực tiếp.",
                )
                if part
            )
        )
    if _looks_multi_intent_question(base):
        subqueries.append(
            "\n".join(
                part
                for part in (
                    base,
                    "Tách ý: định nghĩa, chẩn đoán, phân loại, điều trị, tiên lượng nếu câu hỏi có nhiều ý.",
                )
                if part
            )
        )

    return _unique_texts([item for item in subqueries if item and item != base])[:3]


def _looks_multi_intent_question(query: str) -> bool:
    normalized = _normalize_query_text(query)
    intent_markers = (
        "chan doan",
        "dieu tri",
        "phan loai",
        "trieu chung",
        "tien luong",
        "nguyen nhan",
        "co che",
        "la gi",
    )
    hits = sum(1 for marker in intent_markers if marker in normalized)
    return hits >= 2 or any(separator in normalized for separator in (" va ", " dong thoi ", " kem theo "))


def _grade_and_sort_evidence_items(
    evidence_items: list[dict[str, Any]],
    state: ChatbotGraphState,
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grades: list[dict[str, Any]] = []
    for index, item in enumerate(evidence_items):
        grade = _grade_evidence_item(item, state, index=index)
        enriched_item = {**item, "evidence_grade": grade}
        evidence_items[index] = enriched_item
        grades.append(
            {
                "document_id": item.get("document_id"),
                "source_id": item.get("source_id"),
                **grade,
            }
        )
    ranked = sorted(
        evidence_items,
        key=lambda item: (
            (item.get("evidence_grade") or {}).get("score") or 0.0,
            item.get("fusion_score") or 0.0,
            item.get("keyword_score") or 0.0,
            item.get("similarity") or 0.0,
        ),
        reverse=True,
    )
    sorted_grades = sorted(grades, key=lambda item: item.get("score") or 0.0, reverse=True)
    return ranked[:limit], sorted_grades[:limit]


def _grade_evidence_item(item: dict[str, Any], state: ChatbotGraphState, *, index: int) -> dict[str, Any]:
    content = " ".join(
        str(value or "")
        for value in (
            item.get("content"),
            item.get("preview"),
            item.get("disease_name"),
            item.get("section_type"),
            item.get("biomarker"),
        )
    )
    normalized_content = _normalize_query_text(content)
    normalized_query = _normalize_query_text(state.get("query", ""))
    query_tokens = set(_important_query_tokens(normalized_query))
    context_tokens = set(_important_query_tokens(normalized_content, max_tokens=300))
    token_hits = sorted(query_tokens & context_tokens)

    retrieval_plan = state.get("retrieval_plan") or {}
    soft_hints = retrieval_plan.get("soft_hints") or {}
    term_candidates = [
        *[str(term) for term in soft_hints.get("terms", [])[:8]],
        *[str(term) for term in soft_hints.get("disease_names", [])[:4]],
        *[str(term) for term in soft_hints.get("section_types", [])[:4]],
        *[str(term) for term in soft_hints.get("biomarkers", [])[:4]],
    ]
    term_hits = [
        term
        for term in term_candidates
        if _normalize_query_text(term) and _normalize_query_text(term) in normalized_content
    ]
    metadata_hits = 0
    filters = state.get("filters") or {}
    if filters.get("disease_name") and filters.get("disease_name") == item.get("disease_name"):
        metadata_hits += 1
    if filters.get("section_type") and filters.get("section_type") == item.get("section_type"):
        metadata_hits += 1
    if filters.get("biomarker") and filters.get("biomarker") == item.get("biomarker"):
        metadata_hits += 1

    retrieval_score = float(item.get("fusion_score") or item.get("keyword_score") or item.get("similarity") or 0.0)
    token_score = min(0.35, len(token_hits) * 0.055)
    term_score = min(0.30, len(term_hits) * 0.10)
    metadata_score = min(0.15, metadata_hits * 0.05)
    retrieval_component = min(0.20, retrieval_score)
    score = round(token_score + term_score + metadata_score + retrieval_component, 6)
    return {
        "score": score,
        "token_hits": token_hits[:12],
        "term_hits": term_hits[:8],
        "metadata_hits": metadata_hits,
        "retrieval_score": round(retrieval_score, 6),
        "rank_before_grade": index + 1,
    }


def _should_agentic_retry(state: ChatbotGraphState, quality: dict[str, Any]) -> bool:
    if state.get("route") != "retrieve":
        return False
    if quality.get("sufficient"):
        return False
    if int(state.get("agentic_retry_count") or 0) >= 1:
        return False
    if state.get("medical_tool_result") and state.get("evidence_items"):
        return False
    return bool(state.get("query"))


def _build_agentic_retry_query(state: ChatbotGraphState) -> str:
    retrieval_plan = state.get("retrieval_plan") or {}
    soft_hints = retrieval_plan.get("soft_hints") or {}
    parts = [state.get("query", "")]
    terms = [str(item) for item in soft_hints.get("terms", []) if item][:8]
    diseases = [DISEASE_LABELS.get(str(item), str(item)) for item in soft_hints.get("disease_names", [])][:3]
    sections = [SECTION_LABELS.get(str(item), str(item)) for item in soft_hints.get("section_types", [])][:3]
    biomarkers = [str(item) for item in soft_hints.get("biomarkers", []) if item][:4]
    if diseases:
        parts.append("Bệnh liên quan: " + ", ".join(diseases))
    if sections:
        parts.append("Mục cần tìm: " + ", ".join(sections))
    if biomarkers:
        parts.append("Chỉ số: " + ", ".join(biomarkers))
    if terms:
        parts.append("Từ khóa bắt buộc: " + ", ".join(terms))
    parts.append("Tìm đoạn giải thích trực tiếp, tiêu chuẩn, biểu hiện, phân loại hoặc điều trị liên quan.")
    return "\n".join(part for part in parts if part)


def _merge_agentic_retry_items(
    base_items: list[dict[str, Any]],
    retry_items: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in [*base_items, *retry_items]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("document_id") or item.get("source_id") or len(merged))
        existing = merged.get(key)
        if existing is None or _retrieval_sort_score(item) > _retrieval_sort_score(existing):
            merged[key] = item
    return sorted(merged.values(), key=_retrieval_sort_score, reverse=True)[:limit]


def _unique_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = _normalize_query_text(text)
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _retrieval_sort_score(item: dict[str, Any]) -> float:
    return float(item.get("fusion_score") or item.get("keyword_score") or item.get("similarity") or 0.0)


def _important_query_tokens(normalized_text: str, *, max_tokens: int = 32) -> list[str]:
    stopwords = {
        "benh",
        "than",
        "hoi",
        "chung",
        "viem",
        "cau",
        "nhung",
        "cac",
        "cua",
        "cho",
        "theo",
        "duoc",
        "khi",
        "trong",
        "voi",
        "nguoi",
        "benh",
        "nhan",
        "gi",
        "nao",
    }
    tokens = re.findall(r"[a-z0-9]+", normalized_text)
    return [token for token in tokens if len(token) >= 3 and token not in stopwords][:max_tokens]


def _format_evidence_for_prompt(evidence_items: list[dict[str, Any]]) -> str:
    """Format evidence cho prompt nhưng không đưa số trang, source label hoặc id nội bộ vào."""

    if not evidence_items:
        return "Không tìm thấy evidence phù hợp."

    blocks: list[str] = []
    for item in evidence_items:
        content = (item.get("content") or item.get("preview") or "").strip()
        if content:
            blocks.append(f"<context_item>\n{content[:1800]}\n</context_item>")
    return "\n\n".join(blocks) if blocks else "Không tìm thấy evidence phù hợp."


def _format_web_context(web_results: list[dict[str, Any]]) -> str:
    if not web_results:
        return "Không có ngữ cảnh web y tế bổ sung."

    blocks: list[str] = []
    for item in web_results:
        title = str(item.get("title") or item.get("domain") or "Nguồn y tế").strip()
        domain = str(item.get("domain") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        content = str(item.get("content") or "").strip()
        text = _trim_text(content or snippet, 1600)
        if not text:
            continue
        heading = " - ".join(part for part in (title, domain) if part)
        blocks.append(f"<web_context_item>\n{heading}\n{text}\n</web_context_item>")
    return "\n\n".join(blocks) if blocks else "Không có ngữ cảnh web y tế bổ sung."


def _build_user_sources(
    evidence_items: list[dict[str, Any]],
    web_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
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
    offset = len(sources)
    for index, item in enumerate(web_results or [], start=1):
        sources.append(
            {
                "label": f"Nguồn web y tế {offset + index}",
                "source_type": "medical_web",
                "title": item.get("title"),
                "domain": item.get("domain"),
                "url": item.get("url"),
                "preview": (item.get("snippet") or item.get("content") or "")[:240],
            }
        )
    return sources


def build_user_sources_for_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Public helper for the API streaming path, which stops before cleanup_response."""

    return _build_user_sources(
        state.get("evidence_items", []),
        state.get("web_results", []),
    )


def _web_search_enabled(state: ChatbotGraphState) -> bool:
    requested = state.get("enable_web_search")
    if requested is not None:
        return bool(requested)
    return os.getenv("ENABLE_MEDICAL_WEB_SEARCH", "true").strip().lower() in {"1", "true", "yes", "on"}


def _trim_text(value: str, max_chars: int) -> str:
    value = " ".join(str(value or "").split())
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rsplit(" ", 1)[0] + "..."
