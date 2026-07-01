from __future__ import annotations

"""FastAPI app public cho AI/RAG service của VitalAI."""

import json
import os
import base64
import tempfile
from pathlib import Path
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form
import httpx
import logging
from dotenv import load_dotenv
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
    conversation_id: str | None = None
    user_id: str | None = None
    memory_context: str | None = None
    chat_history: list[dict[str, str]] | None = None
    enable_web_search: bool | None = None


class MemorySummarizeRequest(BaseModel):
    previous_summary: str = ""
    question: str = Field(min_length=1)
    answer: str = Field(default="")


class MemorySummarizeResponse(BaseModel):
    summary: str


class TtsPrepareRequest(BaseModel):
    """Payload tối thiểu để frontend voice mode không bị thiếu endpoint."""
    text: str = Field(default="", description="Text cần đọc bằng TTS phía client.")


class VoiceTranscribeRequest(BaseModel):
    audio_base64: str = Field(..., description="Dữ liệu WAV đã mã hóa base64.")
    language: str = Field(default="vi")


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

# Load .env at startup so downstream services keys are available when running via uvicorn.
load_dotenv()

logger = logging.getLogger(__name__)


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
        "conversation_id": request.conversation_id,
        "user_id": request.user_id,
        "memory_context": request.memory_context,
        "chat_history": request.chat_history,
        "enable_web_search": request.enable_web_search,
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


@app.post("/memory/summarize", response_model=MemorySummarizeResponse)
async def summarize_memory(
    request: MemorySummarizeRequest,
    answerer: Any = Depends(get_answerer),
) -> MemorySummarizeResponse:
    """Tạo rolling summary cho Node backend lưu theo user/session."""

    try:
        summary = await answerer.summarize_memory(
            previous_summary=request.previous_summary,
            question=request.question,
            answer=request.answer,
        )
        return MemorySummarizeResponse(summary=summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể tóm tắt memory: {exc}") from exc


@app.post("/voice/tts/prepare")
async def prepare_tts_text(request: TtsPrepareRequest) -> dict[str, str]:
    """Trả lại text đã trim để voice mode phía frontend có endpoint hợp lệ."""
    # Use centralized TTS text preprocessing
    from src.TTS.tts_handler import prepare_tts_text as _prepare

    speak = _prepare(request.text)
    return {"speak_text": speak, "language": "vi"}


@app.post("/voice/stt")
async def voice_stt(request: VoiceTranscribeRequest) -> dict[str, Any]:
    """Accept base64 WAV -> transcribe using server-side PhoWhisper handler."""
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


@app.post("/health-report/analyze-image")
async def health_report_analyze_image(
    file: UploadFile = File(...),
    language: str = Form("vi"),
    patient_id: Optional[str] = Form(None),
) -> dict[str, Any]:
    """Frontend endpoint: nhận ảnh phiếu khám, gọi tool OCR + evaluate chỉ số."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File tải lên không phải là hình ảnh hợp lệ.")

    medical_tools_url = (os.getenv("MEDICAL_TOOLS_BASE_URL") or "http://127.0.0.1:8010").rstrip("/")

    image_bytes = await file.read()
    filename = file.filename or "upload.png"
    content_type = file.content_type or "image/png"

    form = {"language": language}
    if patient_id is not None:
        form["patient_id"] = patient_id

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{medical_tools_url}/mcp/medical-tools/health-report/analyze-image",
                files={"file": (filename, image_bytes, content_type)},
                data=form,
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Không gọi được medical_tools service: {exc}") from exc

    if resp.status_code == 422:
        detail = resp.json().get("detail", "OCR thất bại.")
        raise HTTPException(status_code=422, detail=detail)
    if not resp.is_success:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    ocr_payload = resp.json()

    eval_payload: dict[str, Any] = {}
    ocr_text = ocr_payload.get("text", "") if isinstance(ocr_payload, dict) else ""
    if ocr_text.strip():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                eval_resp = await client.post(
                    f"{medical_tools_url}/mcp/medical-tools/evaluate",
                    json={"text": ocr_text, "include_debug": False},
                )
            if eval_resp.is_success:
                eval_payload = eval_resp.json()
            else:
                logger.warning("medical_tools.evaluate failed: %s", eval_resp.text[:500])
        except httpx.RequestError as exc:
            logger.warning("medical_tools.evaluate unavailable: %s", exc)

    return {
        "ocr": ocr_payload,
        "evaluation": eval_payload,
    }


def _build_health_report_prompt(
    *,
    user_question: str,
    ocr_text: str,
    evaluation: dict[str, Any],
) -> str:
    matches = evaluation.get("threshold_matches") if isinstance(evaluation, dict) else []
    detected = evaluation.get("detected_measurements") if isinstance(evaluation, dict) else []
    formulas = evaluation.get("formula_results") if isinstance(evaluation, dict) else []

    matches = matches if isinstance(matches, list) else []
    detected = detected if isinstance(detected, list) else []
    formulas = formulas if isinstance(formulas, list) else []

    measured_lines = []
    for item in detected[:20]:
        name = item.get("name", "unknown") if isinstance(item, dict) else "unknown"
        value = item.get("value", "") if isinstance(item, dict) else ""
        unit = f" {item.get('unit')}" if isinstance(item, dict) and item.get("unit") else ""
        source = f" [nguồn: {item.get('source')}]" if isinstance(item, dict) and item.get("source") else ""
        measured_lines.append(f"- {name}: {value}{unit}{source}")

    abnormal_lines = []
    for item in matches:
        if not isinstance(item, dict) or not item.get("matched"):
            continue
        threshold = item.get("threshold") if isinstance(item.get("threshold"), dict) else {}
        label = threshold.get("label")
        if not label:
            continue
        biomarker = item.get("biomarker", "")
        value = item.get("input_value", "")
        unit = item.get("input_unit", "")
        severity = threshold.get("severity")
        sev = f" (mức: {severity})" if severity else ""
        abnormal_lines.append(f"- {biomarker}: {value} {unit} => {label}{sev}".strip())
        if len(abnormal_lines) >= 12:
            break

    formula_lines = []
    for item in formulas:
        if not isinstance(item, dict) or item.get("status") != "computed":
            continue
        name = item.get("output_name") or item.get("formula_id") or "formula"
        value = item.get("value", "")
        unit = f" {item.get('unit')}" if item.get("unit") else ""
        formula_lines.append(f"- {name}: {value}{unit}")
        if len(formula_lines) >= 10:
            break

    return (
        "Bạn là trợ lý y khoa. Người dùng gửi ảnh phiếu khám, dữ liệu đã OCR + parse chỉ số.\n"
        "Hãy NHẬN XÉT CỤ THỂ TỪNG CHỈ SỐ đã liệt kê, không trả lời chung chung.\n\n"
        f"Câu hỏi/yêu cầu của người dùng: {user_question}\n\n"
        "Danh sách chỉ số nhận diện được:\n"
        f"{chr(10).join(measured_lines) if measured_lines else '- (Chưa parse được chỉ số nào)'}\n\n"
        "Chỉ số bất thường/nhãn theo ngưỡng:\n"
        f"{chr(10).join(abnormal_lines) if abnormal_lines else '- (Chưa có chỉ số bất thường tìm được theo ngưỡng)'}\n\n"
        "Kết quả công thức (nếu có):\n"
        f"{chr(10).join(formula_lines) if formula_lines else '- (Không có công thức nào tính được)'}\n\n"
        "OCR gốc (tham chiếu, đã làm sạch):\n"
        f"{(ocr_text or '')[:1200]}\n\n"
        "Yêu cầu trả lời theo format:\n"
        "1) Tóm tắt tổng quan ngắn.\n"
        "2) Nhận xét chi tiết từng chỉ số (mỗi chỉ số 1 dòng: tên chỉ số, trạng thái bình thường/bất thường, ý nghĩa).\n"
        "3) Mức độ ưu tiên theo dõi (thấp/vừa/cao) và lý do.\n"
        "4) Khuyến nghị bước tiếp theo để khám/xét nghiệm bổ sung.\n"
        "5) Cảnh báo: OCR có thể sai và nội dung không thay thế chẩn đoán bác sĩ.\n"
        "Giọng điệu: dễ hiểu, tiếng Việt."
    )


def _build_simple_ocr_prompt(*, user_question: str, ocr_text: str) -> str:
    return (
        "Bạn là trợ lý y khoa. Người dùng gửi ảnh phiếu khám/xét nghiệm.\n"
        f"Câu hỏi: {user_question}\n\n"
        "Nội dung OCR từ ảnh (có thể có lỗi nhận dạng):\n"
        f"{(ocr_text or '').strip()[:4000]}\n\n"
        "Hãy phân tích các chỉ số có trong text, nhận xét bình thường/bất thường nếu đủ thông tin, "
        "khuyến nghị theo dõi và nhắc người dùng đây không thay thế chẩn đoán bác sĩ. Tiếng Việt, dễ hiểu."
    )


@app.post("/health-report/analyze-and-answer")
async def health_report_analyze_and_answer(
    file: UploadFile = File(...),
    question: str = Form("Phân tích ảnh đã tải lên"),
    language: str = Form("vi"),
    patient_id: Optional[str] = Form(None),
    top_k: int = Form(5),
) -> dict[str, Any]:
    """One-shot endpoint: OCR ảnh trước -> Gửi văn bản trích xuất được vào LLM để nhận xét lâm sàng."""
    medical_tools_url = (os.getenv("MEDICAL_TOOLS_BASE_URL") or "http://127.0.0.1:8010").rstrip("/")

    # 1) Đọc file ảnh và gọi dịch vụ medical_tools để thực hiện OCR trích xuất văn bản
    image_bytes = await file.read()
    filename = file.filename or "upload.png"
    content_type = file.content_type or "image/png"

    ocr_payload: dict[str, Any]
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{medical_tools_url}/mcp/medical-tools/health-report/analyze-image",
                files={"file": (filename, image_bytes, content_type)},
                data={"language": language, "patient_id": patient_id} if patient_id is not None else {"language": language},
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Không gọi được dịch vụ medical_tools (OCR): {exc}") from exc

    if resp.status_code == 422:
        detail = resp.json().get("detail", "OCR thất bại.")
        raise HTTPException(status_code=422, detail=detail)
    if not resp.is_success:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    ocr_payload = resp.json()
    ocr_text = ocr_payload.get("text", "") if isinstance(ocr_payload, dict) else ""
    if not ocr_text.strip():
        raise HTTPException(status_code=422, detail="OCR không trích xuất được nội dung văn bản nào từ ảnh.")

    # 2) Đóng gói văn bản OCR và câu hỏi của người dùng vào 1 prompt chuyên biệt
    user_question = question.strip() or "Phân tích ảnh đã tải lên"

    prompt = (
        "Bạn là bác sĩ trợ lý y khoa chuyên nghiệp và chu đáo. Người dùng đã gửi ảnh chụp phiếu kết quả khám sức khỏe, xét nghiệm hoặc kết quả lâm sàng.\n"
        "Dưới đây là toàn bộ nội dung văn bản y khoa trích xuất được từ ảnh (OCR):\n"
        "=========================================\n"
        f"{ocr_text}\n"
        "=========================================\n\n"
        f"Yêu cầu của người dùng: {user_question}\n\n"
        "Nhiệm vụ của bạn:\n"
        "Hãy đọc hiểu văn bản y khoa trên cực kỳ kỹ lưỡng, nhận diện tất cả các chỉ số xét nghiệm, kết quả đo, khoảng tham chiếu và đơn vị có trong văn bản.\n"
        "Sau đó, hãy nhận xét chi tiết, chuyên nghiệp và chính xác về kết quả y tế này bằng tiếng Việt.\n\n"
        "Quy tắc trả lời:\n"
        "1) Trả lời rõ ràng, dễ hiểu, có cấu trúc tốt bằng tiếng Việt (sử dụng định dạng Markdown, bullet points).\n"
        "2) Nhận xét chi tiết từng chỉ số có dấu hiệu bất thường (nằm ngoài khoảng tham chiếu cao/thấp) và giải thích ý nghĩa lâm sàng đơn giản.\n"
        "3) Đưa ra mức độ ưu tiên theo dõi (thấp/vừa/cao) kèm lý do y khoa rõ ràng.\n"
        "4) Đưa ra các khuyến nghị hữu ích về chế độ dinh dưỡng, chế độ sinh hoạt hoặc các xét nghiệm/khám bổ sung tiếp theo nếu cần.\n"
        "5) Luôn kèm theo cảnh báo y khoa: 'Mọi thông tin phân tích từ văn bản ảnh chỉ mang tính chất tham khảo, không thay thế cho chẩn đoán và tư vấn chuyên môn của bác sĩ chuyên khoa.'"
    )

    # 3) Gọi LLM (answerer) để nhận xét
    try:
        answerer = get_answerer()
        llm_result = await answerer.answer(query=prompt, top_k=max(1, min(int(top_k), 20)), include_debug=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tạo nhận xét từ văn bản OCR thất bại: {exc}") from exc

    return {
        "query": question,
        "answer": llm_result.get("answer", ""),
        "route": "ocr_text_qa",
        "sources": [],
        "ocr": ocr_payload,
        "evaluation": {},
    }
