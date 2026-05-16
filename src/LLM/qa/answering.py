from __future__ import annotations

"""
Lớp QA public API cho VitalAI.

Mục tiêu của file này:
- khởi tạo retriever + chat model
- gọi LangGraph chatbot flow
- trả output an toàn cho UI/user, không lộ page number hoặc metadata nội bộ

Giới hạn hiện tại:
- chưa có structured lookup riêng cho threshold/formula
- chưa có long-term memory hoặc tool calling ngoài retrieval
"""

import os
from collections.abc import AsyncIterator
from typing import Any

from langchain_mistralai import ChatMistralAI

from src.LLM.observability import configure_langsmith_from_env
from src.LLM.qa.graph import build_chatbot_graph, cleanup_user_answer
from src.LLM.retrieval.vector_search import NeonVectorSearcher, build_searcher_from_env


class RetrievalAugmentedAnswerer:
    """Điều phối retrieval và answer synthesis bằng Mistral."""

    def __init__(
        self,
        searcher: NeonVectorSearcher,
        llm: ChatMistralAI,
        max_evidence_items: int = 5,
        medical_tools_base_url: str | None = None,
        tool_contract_path: str | None = None,
    ) -> None:
        self.searcher = searcher
        self.llm = llm
        self.max_evidence_items = max_evidence_items
        self.graph = build_chatbot_graph(
            searcher=searcher,
            llm=llm,
            max_evidence_items=max_evidence_items,
            medical_tools_base_url=medical_tools_base_url,
            tool_contract_path=tool_contract_path,
        )

    async def answer(
        self,
        query: str,
        top_k: int = 5,
        disease_name: str | None = None,
        section_type: str | None = None,
        source_type: str | None = None,
        biomarker: str | None = None,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        """Trả về câu trả lời cuối cùng. Debug RAG chỉ bật khi caller yêu cầu."""

        state = await self.graph.ainvoke(
            {
                "query": query,
                "top_k": top_k,
                "disease_name": disease_name,
                "section_type": section_type,
                "source_type": source_type,
                "biomarker": biomarker,
            }
        )
        response: dict[str, Any] = {
            "query": query,
            "answer": state.get("final_answer", ""),
            "route": state.get("route"),
            "sources": state.get("user_sources", []),
        }
        if include_debug:
            response["debug"] = {
                "filters": state.get("filters"),
                "query_understanding": state.get("query_understanding"),
                "results": state.get("debug_results", []),
                "router_plan": state.get("router_plan"),
                "router_error": state.get("router_error"),
                "medical_tool_result": state.get("medical_tool_result"),
                "extracted_tool_payload": state.get("extracted_tool_payload"),
            }
        return response

    async def stream_answer(
        self,
        query: str,
        top_k: int = 5,
        disease_name: str | None = None,
        section_type: str | None = None,
        source_type: str | None = None,
        biomarker: str | None = None,
        include_debug: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream token của final answer và kết thúc bằng event `done`.

        Graph có cả LLM router nội bộ lẫn LLM tạo câu trả lời cuối. Ta chỉ public
        token từ run được tag `final_answer` để UI không bao giờ thấy JSON router.
        """

        graph_input = {
            "query": query,
            "top_k": top_k,
            "disease_name": disease_name,
            "section_type": section_type,
            "source_type": source_type,
            "biomarker": biomarker,
        }
        streamed_text = ""
        final_state: dict[str, Any] | None = None

        async for event in self.graph.astream_events(graph_input, version="v2"):
            event_name = str(event.get("event") or "")
            tags = set(event.get("tags") or [])

            if event_name == "on_chat_model_stream" and "final_answer" in tags:
                token = _chunk_text((event.get("data") or {}).get("chunk"))
                if token:
                    streamed_text += token
                    yield {"event": "token", "token": token}
                continue

            if event_name == "on_chain_end":
                output = (event.get("data") or {}).get("output")
                if isinstance(output, dict) and (
                    "final_answer" in output
                    or "raw_answer" in output
                    or "user_sources" in output
                ):
                    final_state = output

        final_answer = cleanup_user_answer(
            str((final_state or {}).get("final_answer") or streamed_text or "")
        )
        response: dict[str, Any] = {
            "event": "done",
            "query": query,
            "answer": final_answer,
            "route": (final_state or {}).get("route"),
            "sources": (final_state or {}).get("user_sources", []),
        }
        if include_debug:
            response["debug"] = {
                "filters": (final_state or {}).get("filters"),
                "query_understanding": (final_state or {}).get("query_understanding"),
                "results": (final_state or {}).get("debug_results", []),
                "router_plan": (final_state or {}).get("router_plan"),
                "router_error": (final_state or {}).get("router_error"),
                "medical_tool_result": (final_state or {}).get("medical_tool_result"),
                "extracted_tool_payload": (final_state or {}).get("extracted_tool_payload"),
            }
        yield response


def _chunk_text(chunk: Any) -> str:
    """Lấy text từ các shape chunk LangChain thường trả về."""

    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def build_answerer_from_env() -> RetrievalAugmentedAnswerer:
    """Khởi tạo answerer từ `.env` hiện có của dự án."""

    configure_langsmith_from_env()

    api_key = os.getenv("MISTRAL_CLIENT_API_KEY")
    model_name = os.getenv("MODEL_NAME", "mistral-large-latest")
    temperature = float(os.getenv("MISTRAL_TEMPERATURE", "0.1"))
    medical_tools_base_url = os.getenv("MEDICAL_TOOLS_BASE_URL", "http://localhost:8010")
    tool_contract_path = os.getenv("MEDICAL_TOOLS_CONTRACT_PATH")

    if not api_key:
        raise ValueError("Thiếu biến môi trường MISTRAL_CLIENT_API_KEY")

    searcher = build_searcher_from_env()
    llm = ChatMistralAI(
        model=model_name,
        api_key=api_key,
        max_tokens=2048,
        temperature=temperature,
    )
    return RetrievalAugmentedAnswerer(
        searcher=searcher,
        llm=llm,
        medical_tools_base_url=medical_tools_base_url,
        tool_contract_path=tool_contract_path,
    )
