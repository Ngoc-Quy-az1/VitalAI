from __future__ import annotations

from pathlib import Path


DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "tool_contracts" / "medical_tools_contract.md"
ALLOWED_ENDPOINTS = {"/mcp/medical-tools/evaluate"}
ALLOWED_SECTION_TYPES = {
    "definition",
    "classification",
    "clinical_features",
    "diagnosis_criteria",
    "pathology",
    "treatment",
    "progression",
    "complications",
    "follow_up",
    "general",
}
ALLOWED_SOURCE_TYPES = {"chunk", "threshold", "formula"}
ALLOWED_DISEASE_NAMES = {
    "benh_than_man",
    "lupus_nephritis",
    "acute_kidney_injury",
    "hoi_chung_than_hu",
    "benh_than_iga",
    "diabetic_kidney_disease",
    "benh_ly_cau_than",
    "viem_cau_than_cap",
}
DISEASE_NAME_ALIASES = {
    "lupus": "lupus_nephritis",
    "lupus_ban_do": "lupus_nephritis",
    "lupus ban do": "lupus_nephritis",
    "lupus_ban_đỏ": "lupus_nephritis",
    "lupus ban đỏ": "lupus_nephritis",
    "systemic_lupus_erythematosus": "lupus_nephritis",
    "systemic lupus erythematosus": "lupus_nephritis",
    "sle": "lupus_nephritis",
    "benh_than_lupus": "lupus_nephritis",
    "benh than lupus": "lupus_nephritis",
    "viem_than_lupus": "lupus_nephritis",
    "viem than lupus": "lupus_nephritis",
    "benh_than_man_tinh": "benh_than_man",
    "benh than man tinh": "benh_than_man",
    "ckd": "benh_than_man",
    "suy_than_man": "benh_than_man",
    "suy than man": "benh_than_man",
    "aki": "acute_kidney_injury",
    "suy_than_cap": "acute_kidney_injury",
    "suy than cap": "acute_kidney_injury",
}
LABEL_TRANSLATIONS = {
    "prerenal_aki_suggestive": "gợi ý suy thận cấp trước thận",
    "intrinsic_aki_suggestive": "gợi ý suy thận cấp tại thận",
    "anemia_threshold": "ngưỡng thiếu máu",
    "anemia_threshold_female": "ngưỡng thiếu máu ở nữ",
    "hematocrit_low": "hematocrit thấp",
    "blood_pressure_target": "mục tiêu huyết áp",
    "severe_hyperkalemia": "tăng kali máu nặng",
}
