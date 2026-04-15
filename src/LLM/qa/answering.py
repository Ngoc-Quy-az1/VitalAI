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
from typing import Any

from langchain_mistralai import ChatMistralAI

from src.LLM.observability import configure_langsmith_from_env
from src.LLM.qa.graph import build_chatbot_graph
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
            }
        return response


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
