from __future__ import annotations

"""
Lớp QA tối thiểu cho VitalAI: retrieval trước, rồi sinh câu trả lời bằng chat model.

Mục tiêu của file này:
- tái sử dụng hybrid retriever hiện có
- đưa evidence vào một prompt rõ ràng
- sinh câu trả lời tiếng Việt có trích nguồn ngắn gọn

Giới hạn hiện tại:
- chưa có agent graph
- chưa có structured lookup riêng cho threshold/formula
- chưa có self-check hay reranker ở lớp cuối
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI

from src.LLM.retrieval.vector_search import NeonVectorSearcher, build_searcher_from_env


class RetrievalAugmentedAnswerer:
    """Điều phối retrieval và answer synthesis bằng Mistral."""

    def __init__(
        self,
        searcher: NeonVectorSearcher,
        llm: ChatMistralAI,
        max_evidence_items: int = 5,
    ) -> None:
        self.searcher = searcher
        self.llm = llm
        self.max_evidence_items = max_evidence_items

    async def answer(
        self,
        query: str,
        top_k: int = 5,
        disease_name: str | None = None,
        section_type: str | None = None,
        source_type: str | None = None,
        biomarker: str | None = None,
    ) -> dict[str, Any]:
        """Trả về câu trả lời LLM cùng toàn bộ retrieval context để debug."""

        retrieval = await self.searcher.search(
            query=query,
            top_k=top_k,
            disease_name=disease_name,
            section_type=section_type,
            source_type=source_type,
            biomarker=biomarker,
        )
        evidence_items = retrieval["results"][: self.max_evidence_items]
        answer_text = await self._generate_answer(query=query, evidence_items=evidence_items)

        return {
            "query": query,
            "answer": answer_text,
            "filters": retrieval["filters"],
            "query_understanding": retrieval.get("query_understanding"),
            "results": retrieval["results"],
        }

    async def _generate_answer(self, query: str, evidence_items: list[dict[str, Any]]) -> str:
        """Gọi Mistral để tổng hợp câu trả lời từ evidence đã truy xuất."""

        if not evidence_items:
            return (
                "Chưa tìm thấy ngữ cảnh phù hợp trong kho tài liệu hiện tại, "
                "nên mình chưa thể trả lời đáng tin cậy."
            )

        evidence_block = self._format_evidence(evidence_items)
        messages = [
            SystemMessage(
                content=(
                    "Bạn là trợ lý QA y khoa cho VitalAI. "
                    "Chỉ được trả lời dựa trên phần evidence đã cung cấp. "
                    "Nếu evidence chưa đủ, hãy nói rõ là chưa đủ dữ kiện. "
                    "Luôn trả lời bằng tiếng Việt. "
                    "Ưu tiên câu trả lời ngắn gọn, trực tiếp. "
                    "Khi nêu ý chính, thêm citation ngắn ở cuối câu theo dạng [source_id, tr.page]."
                )
            ),
            HumanMessage(
                content=(
                    f"Câu hỏi: {query}\n\n"
                    "Evidence đã truy xuất:\n"
                    f"{evidence_block}\n\n"
                    "Yêu cầu trả lời:\n"
                    "1. Trả lời trực tiếp câu hỏi.\n"
                    "2. Không bịa thêm thông tin ngoài evidence.\n"
                    "3. Nếu có nhiều ý, gộp ngắn gọn và có citation.\n"
                    "4. Nếu evidence mâu thuẫn hoặc chưa đủ, nói rõ giới hạn."
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)
        return str(response.content).strip()

    def _format_evidence(self, evidence_items: list[dict[str, Any]]) -> str:
        """Định dạng evidence cho prompt để model dễ bám nguồn."""

        lines: list[str] = []
        for index, item in enumerate(evidence_items, start=1):
            page = item.get("page")
            page_text = f"tr.{page}" if page is not None else "tr.?"
            similarity = item.get("similarity")
            keyword_score = item.get("keyword_score")
            score_bits = []
            if similarity is not None:
                score_bits.append(f"sim={similarity}")
            if keyword_score is not None:
                score_bits.append(f"fts={keyword_score}")
            score_text = ", ".join(score_bits) if score_bits else "không có score"
            lines.append(
                (
                    f"[{index}] source_id={item['source_id']} | {page_text} | "
                    f"section={item.get('section_type')} | {score_text}\n"
                    f"{item.get('preview', '').strip()}"
                )
            )
        return "\n\n".join(lines)


def build_answerer_from_env() -> RetrievalAugmentedAnswerer:
    """Khởi tạo answerer từ `.env` hiện có của dự án."""

    load_dotenv()

    api_key = os.getenv("MISTRAL_CLIENT_API_KEY")
    model_name = os.getenv("MODEL_NAME", "mistral-large-latest")
    temperature = float(os.getenv("MISTRAL_TEMPERATURE", "0.1"))

    if not api_key:
        raise ValueError("Thiếu biến môi trường MISTRAL_CLIENT_API_KEY")

    searcher = build_searcher_from_env()
    llm = ChatMistralAI(
        model=model_name,
        api_key=api_key,
        max_tokens=1024,
        temperature=temperature,
    )
    return RetrievalAugmentedAnswerer(searcher=searcher, llm=llm)
