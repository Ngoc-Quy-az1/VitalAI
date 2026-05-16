from __future__ import annotations

"""FastAPI app public cho AI/RAG service của VitalAI."""

import json
import os
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class ChatAnswerRequest(BaseModel):

    query: str = Field(min_length=1, description="Câu hỏi người dùng.")
    top_k: int = Field(default=5, ge=1, le=20)
    include_debug: bool = False
    disease_name: str | None = None
    section_type: str | None = None
    source_type: str | None = None
    biomarker: str | None = None


class TtsPrepareRequest(BaseModel):
    """Payload tối thiểu để frontend voice mode không bị thiếu endpoint."""
    text: str = Field(default="", description="Text cần đọc bằng TTS phía client.")


app = FastAPI(
    title="VitalAI AI Service",
    version="1.0.0",
    description="Public AI/RAG API cho chatbot VitalAI.",
)


def cors_origins_from_env() -> list[str]:
    raw = os.getenv("AI_SERVICE_CORS_ORIGINS", "http://localhost:5173")
    return [item.strip() for item in raw.split(",") if item.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_from_env(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_answerer() -> Any:
    """Lazy-load answerer để module API import được ngay cả lúc boot service."""
    from src.LLM.qa.answering import build_answerer_from_env
    return build_answerer_from_env()


def _chat_kwargs(request: ChatAnswerRequest) -> dict[str, Any]:
    query = " ".join(request.query.split())
    if not query:
        raise HTTPException(status_code=422, detail="`query` không được để trống.")
    return {
        "query": query,
        "top_k": request.top_k,
        "disease_name": request.disease_name,
        "section_type": request.section_type,
        "source_type": request.source_type,
        "biomarker": request.biomarker,
        "include_debug": request.include_debug,
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "vitalai-ai", "version": "v1"}


@app.post("/chat/answer")
async def answer_chat(
    request: ChatAnswerRequest,
    answerer: Any = Depends(get_answerer),
) -> dict[str, Any]:
    """Endpoint sync giữ tương thích với frontend/consumer cũ."""

    try:
        return await answerer.answer(**_chat_kwargs(request))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể tạo câu trả lời: {exc}") from exc


@app.post("/chat/stream")
async def stream_chat(
    request: ChatAnswerRequest,
    answerer: Any = Depends(get_answerer),
) -> StreamingResponse:
    """SSE endpoint stream token final answer theo từng chunk từ LLM."""

    kwargs = _chat_kwargs(request)

    async def generate() -> AsyncIterator[str]:
        try:
            async for item in answerer.stream_answer(**kwargs):
                event = str(item.get("event") or "message")
                yield _sse(event, item)
        except Exception as exc:
            yield _sse("error", {"event": "error", "detail": f"Không thể stream câu trả lời: {exc}"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/voice/tts/prepare")
async def prepare_tts_text(request: TtsPrepareRequest) -> dict[str, str]:
    """Trả lại text đã trim để voice mode phía frontend có endpoint hợp lệ."""

    return {"speak_text": " ".join(request.text.split())}
