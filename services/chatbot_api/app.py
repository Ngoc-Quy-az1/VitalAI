from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
import tempfile
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.LLM.qa.answering import RetrievalAugmentedAnswerer, build_answerer_from_env
from src.TTS.tts_handler import prepare_tts_text


class ChatAnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Câu hỏi người dùng.")
    top_k: int = Field(default=5, ge=1, le=20, description="Số lượng chunk retrieval.")
    disease_name: str | None = Field(default=None, description="Filter disease nếu biết.")
    section_type: str | None = Field(default=None, description="Filter section_type nếu biết.")
    source_type: str | None = Field(default=None, description="Filter source_type nếu biết.")
    biomarker: str | None = Field(default=None, description="Filter biomarker nếu biết.")
    include_debug: bool = Field(default=False, description="Trả thêm debug retrieval/router.")


class ChatAnswerResponse(BaseModel):
    query: str
    answer: str
    route: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    debug: dict[str, Any] | None = None


class VoiceTranscribeResponse(BaseModel):
    text: str
    language: str = "vi"


class VoiceTranscribeRequest(BaseModel):
    audio_base64: str = Field(..., description="Dữ liệu WAV đã mã hóa base64.")
    language: str = Field(default="vi")


class VoiceTtsPrepareRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Nội dung chatbot cần đọc.")


class VoiceTtsPrepareResponse(BaseModel):
    speak_text: str
    language: str = "vi"


app = FastAPI(
    title="VitalAI Chatbot QA API",
    version="1.0.0",
    description="HTTP API cho luồng hỏi đáp chatbot (RAG + tool routing).",
)


@lru_cache(maxsize=1)
def get_answerer() -> RetrievalAugmentedAnswerer:
    """Cache answerer để tái sử dụng kết nối model/retriever."""
    return build_answerer_from_env()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "vitalai-chatbot-api", "version": "v1"}


@app.post("/chat/answer", response_model=ChatAnswerResponse)
async def chat_answer(request: ChatAnswerRequest) -> dict[str, Any]:
    try:
        answerer = get_answerer()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Khởi tạo chatbot thất bại: {exc}") from exc

    try:
        result = await answerer.answer(
            query=request.query,
            top_k=request.top_k,
            disease_name=request.disease_name,
            section_type=request.section_type,
            source_type=request.source_type,
            biomarker=request.biomarker,
            include_debug=request.include_debug,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Xử lý câu hỏi thất bại: {exc}") from exc
    return result


@app.post("/chat/ask", response_model=ChatAnswerResponse)
async def chat_ask(request: ChatAnswerRequest) -> dict[str, Any]:
    """Alias endpoint cho client cũ."""
    return await chat_answer(request)


@app.post("/voice/stt", response_model=VoiceTranscribeResponse)
async def voice_stt(request: VoiceTranscribeRequest) -> dict[str, Any]:
    suffix = ".wav"
    try:
        data = base64.b64decode(request.audio_base64)
        from src.STT.wav_stt_helpers import transcribe_wav_file

        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            text = transcribe_wav_file(Path(tmp.name))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"STT thất bại: {exc}") from exc
    return {"text": text, "language": request.language}


@app.post("/voice/tts/prepare", response_model=VoiceTtsPrepareResponse)
async def voice_tts_prepare(request: VoiceTtsPrepareRequest) -> dict[str, Any]:
    """Chuẩn hóa text để frontend/browser đọc tự nhiên hơn."""
    speak_text = prepare_tts_text(request.text)
    return {"speak_text": speak_text, "language": "vi"}

