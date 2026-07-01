from __future__ import annotations

"""FastAPI app cho structured medical tools.

Service này được thiết kế để deploy riêng với AI/RAG service. AI service hoặc MCP
adapter có thể gọi HTTP endpoints ở đây khi phát hiện input có chỉ số xét nghiệm.
"""

import os
import tempfile
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.OCR.mistral_ocr import MistralOcrError, run_mistral_ocr
from services.medical_tools.service import MedicalToolsService

# Load .env so OCR tool and thresholds share the same config as chatbot.
load_dotenv()


class StructuredEvaluateRequest(BaseModel):
    """Request body cho threshold/formula tool."""

    text: str | None = Field(default=None, description="Text người dùng, ví dụ: 'ACR 350 mg/g, GFR 55'.")
    measurements: Any = Field(
        default=None,
        description="Dict hoặc list chỉ số đã parse sẵn. Ví dụ {'ACR': {'value': 350, 'unit': 'mg/g'}}.",
    )
    disease_name: str | None = Field(default=None, description="Filter disease_name nếu caller đã biết context bệnh.")
    formula_ids: list[str] = Field(default_factory=list, description="Chỉ chạy các formula_id cụ thể nếu cần.")
    include_debug: bool = Field(default=False, description="Trả thêm thông tin debug nội bộ cho developer.")

class GraphQueryRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi cần truy vấn trên dữ liệu sơ đồ.")
    document_id: str | None = Field(default=None, description="Graph document_id nếu muốn khóa vào 1 sơ đồ cụ thể.")
    top_k: int = Field(default=3, ge=1, le=10)


class StructuredKnowledgeQueryRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi liên quan bảng/sơ đồ cần truy xuất structured.")
    top_k: int = Field(default=5, ge=1, le=10)


app = FastAPI(
    title="VitalAI Medical Tools Service",
    version="1.0.0",
    description="Structured threshold/formula API để MCP hoặc AI service gọi độc lập.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_service() -> MedicalToolsService:
    data_dir = os.getenv("MEDICAL_TOOLS_DATA_DIR", "data/processed_data")
    return MedicalToolsService(processed_data_dir=data_dir)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "vitalai-medical-tools", "version": "v1"}


@app.get("/mcp/capabilities")
@app.get("/structured/capabilities")
def capabilities() -> dict[str, Any]:
    return get_service().capabilities()


@app.post("/mcp/medical-tools/evaluate")
@app.post("/structured/evaluate")
def evaluate_structured_input(request: StructuredEvaluateRequest) -> dict[str, Any]:
    return get_service().evaluate(
        text=request.text,
        measurements=request.measurements,
        disease_name=request.disease_name,
        formula_ids=request.formula_ids,
        include_debug=request.include_debug,
    )


@app.post("/mcp/medical-tools/graph-query")
@app.post("/structured/graph-query")
def graph_query(request: GraphQueryRequest) -> dict[str, Any]:
    return get_service().graph_query(
        query=request.query,
        document_id=request.document_id,
        top_k=request.top_k,
    )


@app.post("/mcp/medical-tools/structured-knowledge-query")
@app.post("/structured/knowledge-query")
def structured_knowledge_query(request: StructuredKnowledgeQueryRequest) -> dict[str, Any]:
    return get_service().query_structured_knowledge(
        query=request.query,
        top_k=request.top_k,
    )

@app.post("/health-report/analyze-image")
@app.post("/mcp/medical-tools/health-report/analyze-image")
@app.post("/mcp/medical-tools/tools/health-report-ocr")
async def health_report_analyze_image(
    file: UploadFile = File(...),
    language: str = Form("vi"),
    patient_id: Optional[str] = Form(None),
) -> dict[str, Any]:
    """Endpoint/tool MCP: OCR phiếu khám bằng Mistral và trả text."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File tải lên không phải là hình ảnh hợp lệ.")

    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    temp_file_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_file_path = tmp.name
            buffer = tmp
            shutil.copyfileobj(file.file, buffer)

        ocr_result = run_mistral_ocr(image_path=Path(temp_file_path), language=language)
        return {
            "tool_name": "health_report_ocr",
            "text": ocr_result["text"],
            "raw_text": ocr_result.get("raw_text", ""),
            "filename": file.filename,
            "language": language,
            "patient_id": patient_id,
            "model": ocr_result.get("model"),
        }
    except MistralOcrError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh phiếu khám: {str(e)}") from e
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)