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
from src.LLM.prompts.templates import DIRECT_ANSWER_PROMPT, RAG_ANSWER_PROMPT
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
        try:
            tool_contract = load_medical_tools_contract(tool_contract_path)
            prompt = MEDICAL_TOOL_ROUTER_PROMPT.invoke(
                {
                    "tool_contract": tool_contract,
                    "query": state["query"],
                }
            )
            response = await llm.ainvoke(prompt.messages)
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

        retrieval = await searcher.search(
            query=retrieval_query,
            top_k=state["top_k"],
            disease_name=disease_name,
            section_type=section_type,
            source_type=source_type,
            biomarker=biomarker,
        )
        evidence_items = retrieval["results"][:max_evidence_items]
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

    async def generate_response(state: ChatbotGraphState) -> ChatbotGraphState:
        has_structured_result = bool(state.get("medical_tool_result"))
        if state.get("route") == "retrieve" and not state.get("evidence_items") and not has_structured_result:
            raw_answer = (
                "Mình chưa tìm thấy ngữ cảnh phù hợp trong kho tài liệu hiện tại, "
                "nên chưa thể trả lời chắc chắn cho câu hỏi này."
            )
        elif state.get("route") == "retrieve" and not state.get("evidence_items") and has_structured_result:
            raw_answer = build_structured_answer(
                state.get("medical_tool_result"),
                query=state.get("query"),
            ) or (
                "Mình đã nhận được chỉ số bạn cung cấp, nhưng chưa có đủ kết quả ngưỡng, "
                "phân loại hoặc công thức phù hợp để trả lời chắc chắn."
            )
        else:
            response = await llm.ainvoke(state["prompt_messages"])
            raw_answer = str(response.content).strip()
        return {**state, "raw_answer": raw_answer}

    async def cleanup_response(state: ChatbotGraphState) -> ChatbotGraphState:
        final_answer = cleanup_user_answer(state.get("raw_answer", ""))
        return {
            **state,
            "final_answer": final_answer,
            "user_sources": _build_user_sources(state.get("evidence_items", [])),
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
    graph.add_edge("retrieve_context", "build_prompt")
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


def _is_direct_query(query: str) -> bool:
    """Routing nhẹ: chỉ bỏ qua RAG cho lời chào/cảm ơn/hỏi khả năng hệ thống."""

    direct_patterns = (
        "xin chào",
        "chào bạn",
        "hello",
        "hi ",
        "cảm ơn",
        "cam on",
        "bạn là ai",
        "ban la ai",
        "bạn làm được gì",
        "ban lam duoc gi",
    )
    return any(pattern in f"{query} " for pattern in direct_patterns)


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
