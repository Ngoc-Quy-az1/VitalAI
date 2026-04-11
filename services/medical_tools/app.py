from __future__ import annotations

"""FastAPI app cho structured medical tools.

Service này được thiết kế để deploy riêng với AI/RAG service. AI service hoặc MCP
adapter có thể gọi HTTP endpoints ở đây khi phát hiện input có chỉ số xét nghiệm.
"""

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.medical_tools.service import MedicalToolsService


class StructuredEvaluateRequest(BaseModel):
    """Request body cho threshold/formula tool."""

    text: str | None = Field(default=None, description="Text người dùng, ví dụ: 'ACR 350 mg/g, GFR 55'.")
    measurements: Any = Field(
        default=None,
        description="Dict hoặc list chỉ số đã parse sẵn. Ví dụ {'ACR': {'value': 350, 'unit': 'mg/g'}}.",
    )
    disease_name: str | None = Field(default=None, description="Filter disease_name nếu caller đã biết context bệnh.")
    formula_ids: list[str] | None = Field(default=None, description="Chỉ chạy các formula_id cụ thể nếu cần.")
    include_debug: bool = Field(default=False, description="Trả thêm thông tin debug nội bộ cho developer.")


app = FastAPI(
    title="VitalAI Medical Tools Service",
    version="1.0.0",
    description="Structured threshold/formula API để MCP hoặc AI service gọi độc lập.",
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
