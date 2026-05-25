from __future__ import annotations

"""Prepare richer VitalAI evaluation datasets for LangSmith.

This script does two repeatable data-prep jobs:

1. Enrich `tests/cases/*.json` with reference answers, required facts, and
   gold document/source IDs for workflow/tool regression evaluation.
2. Convert `data/evaluate_data/qa_dataset_50_questions.json` into a
   LangSmith-ready RAG QA dataset with matched source evidence from the
   processed medical corpus.
"""

import argparse
import json
import re
import unicodedata
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "tests" / "cases"
EVAL_DATA_DIR = ROOT / "data" / "evaluate_data"
PROCESSED_DATA_DIR = ROOT / "data" / "processed_data"

RAW_QA_PATH = EVAL_DATA_DIR / "qa_dataset_50_questions.json"
ENRICHED_QA_PATH = EVAL_DATA_DIR / "qa_dataset_50_questions_enriched.json"
LANGSMITH_JSON_PATH = EVAL_DATA_DIR / "langsmith_rag_dataset.json"
LANGSMITH_JSONL_PATH = EVAL_DATA_DIR / "langsmith_rag_dataset.jsonl"

VI_STOPWORDS = {
    "a",
    "ai",
    "anh",
    "bao",
    "bằng",
    "bị",
    "các",
    "cái",
    "cần",
    "cho",
    "có",
    "của",
    "đã",
    "được",
    "gì",
    "hay",
    "khi",
    "là",
    "làm",
    "một",
    "nào",
    "như",
    "những",
    "ở",
    "ra",
    "sao",
    "theo",
    "thì",
    "trong",
    "và",
    "về",
    "với",
}


CASE_ENRICHMENTS: dict[str, dict[str, Any]] = {
    "general_001": {
        "reference_answer": (
            "Lupus ban đỏ hệ thống là bệnh tự miễn có biểu hiện đa cơ quan. Trong tài liệu thận, "
            "bệnh thận lupus/viêm thận lupus là tổn thương phổ biến và quan trọng; bệnh nhân lupus "
            "cần được tầm soát tổn thương thận vì tổn thương thận ảnh hưởng nhiều đến tiên lượng."
        ),
        "required_facts": [
            "Lupus ban đỏ hệ thống là một bệnh tự miễn ảnh hưởng đến nhiều hệ thống cơ quan.",
            "Tổn thương thận là nguyên nhân quan trọng dẫn đến tử vong trong lupus ban đỏ hệ thống.",
            "Bệnh thận lupus hay viêm thận lupus là tổn thương phổ biến và quan trọng.",
            "Bệnh nhân lupus cần được tầm soát tổn thương thận.",
        ],
        "relevant_document_ids": [
            "chunk::lupus_nephritis_p25_002",
            "chunk::lupus_nephritis_p25_003",
            "chunk::lupus_nephritis_p25_004",
            "chunk::lupus_nephritis_p27_003",
        ],
        "relevant_source_ids": [
            "lupus_nephritis_p25_002",
            "lupus_nephritis_p25_003",
            "lupus_nephritis_p25_004",
            "lupus_nephritis_p27_003",
        ],
        "eval_tags": ["general", "definition", "lupus_nephritis", "rag_only"],
        "eval_notes": "Cần trả lời khái niệm lupus, không tự mở rộng sang phác đồ điều trị.",
    },
    "general_002": {
        "reference_answer": (
            "GFR là mức lọc cầu thận, dùng để đánh giá chức năng thận và phân giai đoạn bệnh thận mạn. "
            "Trong tài liệu, GFR < 60 ml/ph/1,73m2 kéo dài trên 3 tháng là tiêu chuẩn chẩn đoán bệnh thận mạn; "
            "GFR có thể được ước tính bằng công thức như MDRD dựa trên creatinin huyết thanh, tuổi, giới và chủng tộc."
        ),
        "required_facts": [
            "GFR là mức lọc cầu thận.",
            "GFR được dùng để đánh giá chức năng thận và phân giai đoạn bệnh thận mạn.",
            "GFR <60 ml/ph/1,73m2 kéo dài trên 3 tháng là tiêu chuẩn chẩn đoán bệnh thận mạn.",
            "GFR có thể được ước tính bằng công thức MDRD dựa trên creatinin huyết thanh, tuổi, giới và chủng tộc.",
        ],
        "relevant_document_ids": [
            "chunk::benh_than_man_p133_002",
            "chunk::benh_than_man_p133_003",
            "chunk::benh_than_man_p141_002",
            "chunk::benh_than_man_p141_003",
        ],
        "relevant_source_ids": [
            "benh_than_man_p133_002",
            "benh_than_man_p133_003",
            "benh_than_man_p141_002",
            "benh_than_man_p141_003",
        ],
        "eval_tags": ["general", "definition", "gfr", "ckd", "rag_only"],
        "eval_notes": "Cần nêu được vai trò đánh giá chức năng thận và mốc <60 kéo dài trên 3 tháng.",
    },
    "general_003": {
        "reference_answer": (
            "ACR là tỷ lệ albumin/creatinin niệu, dùng để đánh giá albumin niệu/tổn thương thận. "
            "Theo phân loại KDIGO trong dữ liệu, A1 là <30 mg/g, A2 là 30-300 mg/g, A3 là >=300 mg/g "
            "hoặc >300 mg/g tùy đoạn nguồn, với nguy cơ bệnh thận mạn tăng dần."
        ),
        "required_facts": [
            "ACR là tỷ lệ albumin/creatinin niệu.",
            "ACR >30 mg/g là biểu hiện albumin niệu trong tiêu chuẩn tổn thương thận.",
            "A1 là ACR <30 mg/g.",
            "A2 là ACR 30-300 mg/g.",
            "A3 là ACR >=300 mg/g hoặc >300 mg/g theo nguồn phân loại.",
        ],
        "relevant_document_ids": [
            "chunk::benh_than_man_p129_001",
            "chunk::benh_than_man_p202_005",
            "threshold::benh_than_man_acr_gt_30_p129_101",
            "threshold::benh_than_man_acr_gteq_300_0_p201_003",
        ],
        "relevant_source_ids": [
            "benh_than_man_p129_001",
            "benh_than_man_p202_005",
            "benh_than_man_acr_gt_30_p129_101",
            "benh_than_man_acr_gteq_300_0_p201_003",
        ],
        "eval_tags": ["general", "definition", "acr", "albuminuria", "rag_only"],
        "eval_notes": "Không cần tool vì không có giá trị ACR cụ thể, nhưng context nên có phân loại A1/A2/A3.",
    },
    "general_004": {
        "reference_answer": (
            "FENa là phân suất bài tiết natri, được tính từ natri niệu, natri máu, creatinin niệu và creatinin máu. "
            "Trong tổn thương thận cấp, FENa <1% gợi ý suy thận trước thận; FENa >1% gợi ý suy thận tại thận."
        ),
        "required_facts": [
            "FENa là Fractional Excretion of Sodium.",
            "FENa dùng natri niệu, natri máu, creatinin niệu và creatinin máu.",
            "FENa <1% gợi ý suy thận trước thận.",
            "FENa >1% gợi ý suy thận tại thận.",
        ],
        "relevant_document_ids": [
            "formula::fena_formula",
            "chunk::acute_kidney_injury_p186_003",
            "threshold::extra_acute_kidney_injury_fena_lt_1_0_p186_025",
            "threshold::extra_acute_kidney_injury_fena_gt_1_0_p186_026",
        ],
        "relevant_source_ids": [
            "fena_formula",
            "acute_kidney_injury_p186_003",
            "extra_acute_kidney_injury_fena_lt_1_0_p186_025",
            "extra_acute_kidney_injury_fena_gt_1_0_p186_026",
        ],
        "eval_tags": ["general", "definition", "fena", "aki", "formula_knowledge"],
        "eval_notes": "Câu hỏi định nghĩa nên không bắt buộc tính toán, nhưng answer phải nêu đúng ý nghĩa ngưỡng.",
    },
    "threshold_001": {
        "reference_answer": (
            "ACR 350 mg/g vượt ngưỡng 300 mg/g nên được xếp nhóm A3, nghĩa là albumin niệu tăng nặng/nguy cơ bệnh thận mạn tăng. "
            "Kết quả này cần được diễn giải cùng bối cảnh lâm sàng và không thay thế đánh giá của bác sĩ."
        ),
        "required_facts": [
            "ACR 350 mg/g >= 300 mg/g.",
            "ACR >= 300 mg/g thuộc nhóm A3.",
            "A3 tương ứng albumin niệu tăng nặng hoặc nguy cơ bệnh thận mạn tăng.",
        ],
        "relevant_document_ids": [
            "threshold::benh_than_man_acr_gteq_300_0_p201_003",
            "chunk::benh_than_man_p202_005",
            "threshold::benh_than_man_acr_gt_30_p129_101",
        ],
        "relevant_source_ids": [
            "benh_than_man_acr_gteq_300_0_p201_003",
            "benh_than_man_p202_005",
            "benh_than_man_acr_gt_30_p129_101",
        ],
        "eval_tags": ["threshold", "acr", "a3", "medical_tool", "classification"],
        "eval_notes": "Tool phải trả classification A3 và answer không được chỉ nói chung chung.",
    },
    "threshold_002": {
        "reference_answer": (
            "GFR 55 ml/ph/1,73m2 nằm trong khoảng 45-60 nên thuộc nhóm G3a; theo phân loại 5 giai đoạn cũ, "
            "30-60 cũng tương ứng CKD stage III. Nếu GFR <60 kéo dài trên 3 tháng, đây là tiêu chuẩn bệnh thận mạn."
        ),
        "required_facts": [
            "GFR 55 ml/ph/1,73m2 nằm trong khoảng 45 đến dưới 60.",
            "45 <= GFR < 60 thuộc nhóm G3a.",
            "30 <= GFR < 60 tương ứng CKD stage III.",
            "GFR <60 kéo dài trên 3 tháng là tiêu chuẩn chẩn đoán bệnh thận mạn.",
        ],
        "relevant_document_ids": [
            "threshold::benh_than_man_gfr_between_45_0_p150_001",
            "threshold::benh_than_man_gfr_between_30_0_p143_003",
            "chunk::benh_than_man_p133_002",
        ],
        "relevant_source_ids": [
            "benh_than_man_gfr_between_45_0_p150_001",
            "benh_than_man_gfr_between_30_0_p143_003",
            "benh_than_man_p133_002",
        ],
        "eval_tags": ["threshold", "gfr", "g3a", "ckd", "medical_tool", "classification"],
        "eval_notes": "Cần phân biệt G3a với CKD stage III, không kết luận bệnh thận mạn nếu thiếu yếu tố kéo dài 3 tháng.",
    },
    "threshold_003": {
        "reference_answer": (
            "ACR 300 mg/g nằm đúng mốc vào nhóm A3 vì rule trong tool là ACR >= 300 thì A3. "
            "Không nên xếp A2 vì A2 chỉ từ 30 đến dưới 300 mg/g."
        ),
        "required_facts": [
            "ACR 300 mg/g đạt điều kiện >= 300 mg/g.",
            "ACR >= 300 mg/g thuộc A3.",
            "A2 là 30 <= ACR < 300 mg/g.",
        ],
        "relevant_document_ids": [
            "threshold::benh_than_man_acr_gteq_300_0_p201_003",
            "threshold::benh_than_man_acr_between_30_0_p201_002",
        ],
        "relevant_source_ids": [
            "benh_than_man_acr_gteq_300_0_p201_003",
            "benh_than_man_acr_between_30_0_p201_002",
        ],
        "eval_tags": ["threshold", "acr", "boundary", "a3", "medical_tool", "classification"],
        "eval_notes": "Boundary case: 300 phải là A3.",
    },
    "threshold_004": {
        "reference_answer": "GFR 75 ml/ph/1,73m2 nằm trong khoảng 60 đến dưới 90 nên thuộc nhóm G2; trong bảng 5 giai đoạn cũ cũng tương ứng CKD stage II.",
        "required_facts": [
            "GFR 75 nằm trong khoảng 60 đến dưới 90.",
            "60 <= GFR < 90 thuộc nhóm G2.",
            "60 <= GFR < 90 cũng tương ứng CKD stage II trong phân loại cũ.",
        ],
        "relevant_document_ids": [
            "threshold::benh_than_man_gfr_between_60_0_p149_002",
            "threshold::benh_than_man_gfr_between_60_0_p143_002",
        ],
        "relevant_source_ids": [
            "benh_than_man_gfr_between_60_0_p149_002",
            "benh_than_man_gfr_between_60_0_p143_002",
        ],
        "eval_tags": ["threshold", "gfr", "g2", "medical_tool", "classification"],
        "eval_notes": "Answer phải có G2, không được nhầm sang G3.",
    },
    "threshold_005": {
        "reference_answer": "GFR 10 ml/ph/1,73m2 nhỏ hơn 15 nên thuộc nhóm G5; theo bảng giai đoạn CKD cũng tương ứng CKD stage V/giai đoạn cuối.",
        "required_facts": [
            "GFR 10 nhỏ hơn 15.",
            "GFR <15 thuộc nhóm G5.",
            "GFR <15 tương ứng CKD stage V.",
        ],
        "relevant_document_ids": [
            "threshold::benh_than_man_gfr_lt_15_0_p150_004",
            "threshold::benh_than_man_gfr_lt_15_0_p144_002",
            "threshold::benh_than_man_gfr_lt_15_p170_1301",
        ],
        "relevant_source_ids": [
            "benh_than_man_gfr_lt_15_0_p150_004",
            "benh_than_man_gfr_lt_15_0_p144_002",
            "benh_than_man_gfr_lt_15_p170_1301",
        ],
        "eval_tags": ["threshold", "gfr", "g5", "ckd_stage_v", "medical_tool", "classification"],
        "eval_notes": "Cần trả lời mạnh vào phân nhóm G5/stage V, không tự tư vấn điều trị thay thế thận nếu không có hỏi.",
    },
    "threshold_006": {
        "reference_answer": (
            "Kali 6.5 mmol/L đạt ngưỡng tăng kali máu nặng trong tài liệu tổn thương thận cấp. "
            "Nguồn nêu kali máu >=6,5 mmol/L kèm biến đổi điện tim là chỉ định cần xử trí/lọc máu sớm trong bối cảnh phù hợp."
        ),
        "required_facts": [
            "Kali 6.5 mmol/L đạt ngưỡng >=6.5 mmol/L.",
            "Ngưỡng >=6.5 mmol/L được gắn nhãn severe_hyperkalemia.",
            "Tài liệu nêu tăng kali máu kèm biến đổi điện tim là dấu hiệu nguy hiểm.",
        ],
        "relevant_document_ids": [
            "threshold::extra_acute_kidney_injury_potassium_gte_6_5_p126_005",
            "threshold::extra_acute_kidney_injury_potassium_gte_6_5_p194_030",
            "chunk::acute_kidney_injury_p194_001",
        ],
        "relevant_source_ids": [
            "extra_acute_kidney_injury_potassium_gte_6_5_p126_005",
            "extra_acute_kidney_injury_potassium_gte_6_5_p194_030",
            "acute_kidney_injury_p194_001",
        ],
        "eval_tags": ["threshold", "potassium", "hyperkalemia", "boundary", "medical_tool", "safety"],
        "eval_notes": "Cần nhấn mạnh nguy hiểm nhưng không kê toa; có thể khuyến nghị đánh giá y tế khẩn nếu có triệu chứng/ECG bất thường.",
    },
    "threshold_007": {
        "reference_answer": (
            "Huyết áp 128/78 mmHg thấp hơn mục tiêu <130/80 mmHg nên đạt mục tiêu trong bối cảnh bệnh thận mạn/bệnh thận đái tháo đường được tài liệu nêu. "
            "Cần hiểu đây là so sánh theo ngưỡng, còn mục tiêu cá thể hóa phụ thuộc bác sĩ điều trị."
        ),
        "required_facts": [
            "Huyết áp tâm thu 128 mmHg < 130 mmHg.",
            "Huyết áp tâm trương 78 mmHg < 80 mmHg.",
            "Mục tiêu huyết áp trong nguồn là <130/80 mmHg.",
        ],
        "relevant_document_ids": [
            "threshold::extra_benh_than_man_systolic_bp_lt_130_0_p75_001",
            "threshold::extra_benh_than_man_diastolic_bp_lt_80_0_p75_002",
            "chunk::benh_than_man_p75_007",
            "chunk::diabetic_kidney_disease_p204_001",
            "chunk::diabetic_kidney_disease_p210_005",
        ],
        "relevant_source_ids": [
            "extra_benh_than_man_systolic_bp_lt_130_0_p75_001",
            "extra_benh_than_man_diastolic_bp_lt_80_0_p75_002",
            "benh_than_man_p75_007",
            "diabetic_kidney_disease_p204_001",
            "diabetic_kidney_disease_p210_005",
        ],
        "eval_tags": ["threshold", "blood_pressure", "target", "medical_tool", "parser"],
        "eval_notes": "Cần parse được cả systolic và diastolic từ dạng 128/78.",
    },
    "threshold_008": {
        "reference_answer": (
            "Nam có Hb 12 g/dL tương đương 120 g/L. Theo ngưỡng WHO trong dữ liệu, nam thiếu máu khi Hb <13 g/dL "
            "(tức <130 g/L), vì vậy kết quả này được xem là thiếu máu theo ngưỡng nam."
        ),
        "required_facts": [
            "Hb 12 g/dL tương đương 120 g/L.",
            "Nam thiếu máu khi Hb <13 g/dL hoặc <130 g/L theo nguồn WHO.",
            "Hb 12 g/dL ở nam thấp hơn ngưỡng 13 g/dL.",
        ],
        "relevant_document_ids": [
            "threshold::extra_benh_than_man_hemoglobin_lt_130_0_p155_012",
            "chunk::benh_than_man_p155_002",
        ],
        "relevant_source_ids": [
            "extra_benh_than_man_hemoglobin_lt_130_0_p155_012",
            "benh_than_man_p155_002",
        ],
        "eval_tags": ["threshold", "hemoglobin", "anemia", "sex_specific", "unit_conversion"],
        "eval_notes": "Cần tôn trọng sex=male và chuyển/hiểu g/dL sang g/L.",
    },
    "threshold_009": {
        "reference_answer": (
            "Albumin máu 2.8 g/dL tương đương 28 g/L, thấp hơn ngưỡng 30 g/L trong hội chứng thận hư/bệnh lý cầu thận, "
            "vì vậy là giảm albumin máu theo ngưỡng dữ liệu."
        ),
        "required_facts": [
            "Albumin máu 2.8 g/dL tương đương 28 g/L.",
            "Albumin máu <30 g/L là giảm theo nguồn hội chứng thận hư/bệnh lý cầu thận.",
            "28 g/L thấp hơn 30 g/L.",
        ],
        "relevant_document_ids": [
            "threshold::hoi_chung_than_hu_albumin_mau_lt_30_p1_903",
            "threshold::benh_ly_cau_than_albumin_mau_lt_30_p19_2002",
            "threshold::hoi_chung_than_hu_albumin_mau_lt_30_p20_602",
            "chunk::benh_ly_cau_than_p1_006",
            "chunk::benh_ly_cau_than_p19_007",
            "chunk::hoi_chung_than_hu_p20_006",
        ],
        "relevant_source_ids": [
            "hoi_chung_than_hu_albumin_mau_lt_30_p1_903",
            "benh_ly_cau_than_albumin_mau_lt_30_p19_2002",
            "hoi_chung_than_hu_albumin_mau_lt_30_p20_602",
            "benh_ly_cau_than_p1_006",
            "benh_ly_cau_than_p19_007",
            "hoi_chung_than_hu_p20_006",
        ],
        "eval_tags": ["threshold", "albumin", "unit_conversion", "medical_tool"],
        "eval_notes": "Đây là case chuyển đơn vị g/dL -> g/L.",
    },
    "formula_001": {
        "reference_answer": (
            "Với nữ 60 tuổi, creatinine 1.4 mg/dL, CKD-EPI 2021 creatinine ước tính eGFR khoảng 43.07 ml/ph/1.73m2. "
            "Giá trị này rơi vào G3b vì nằm trong khoảng 30 đến dưới 45 ml/ph/1.73m2."
        ),
        "required_facts": [
            "CKD-EPI 2021 creatinine dùng tuổi, giới và creatinine huyết thanh.",
            "Kết quả eGFR xấp xỉ 43.07 ml/ph/1.73m2.",
            "eGFR 43.07 nằm trong khoảng 30 đến dưới 45.",
            "30 <= GFR < 45 thuộc nhóm G3b.",
        ],
        "relevant_document_ids": [
            "formula::ckd_epi_2021_creatinine",
            "threshold::benh_than_man_gfr_between_30_0_p150_002",
            "chunk::benh_than_man_p141_003",
        ],
        "relevant_source_ids": [
            "ckd_epi_2021_creatinine",
            "benh_than_man_gfr_between_30_0_p150_002",
            "benh_than_man_p141_003",
        ],
        "eval_tags": ["formula", "ckd_epi_2021", "gfr", "g3b", "medical_tool"],
        "eval_notes": "CKD-EPI có trong formulas.json nhưng chưa chắc nằm trong embedding_documents cũ; evaluator cần tính cả tool context.",
    },
    "formula_006": {
        "reference_answer": (
            "Với nữ 60 tuổi, creatinine 1.4 mg/dL, MDRD eGFR khoảng 40.77 ml/ph/1.73m2 nếu giả định race=other. "
            "Kết quả nằm trong khoảng 30 đến dưới 45 nên thuộc G3b."
        ),
        "required_facts": [
            "MDRD dùng creatinine huyết thanh, tuổi, giới và chủng tộc.",
            "Khi không cung cấp race, tool giả định race=other.",
            "Kết quả MDRD xấp xỉ 40.77 ml/ph/1.73m2.",
            "30 <= GFR < 45 thuộc nhóm G3b.",
        ],
        "relevant_document_ids": [
            "formula::mdrd_gfr",
            "chunk::benh_than_man_p133_003",
            "threshold::benh_than_man_gfr_between_30_0_p150_002",
        ],
        "relevant_source_ids": [
            "mdrd_gfr",
            "benh_than_man_p133_003",
            "benh_than_man_gfr_between_30_0_p150_002",
        ],
        "eval_tags": ["formula", "mdrd", "gfr", "default_race", "medical_tool"],
        "eval_notes": "Answer nên nêu giả định race=other nếu dùng MDRD.",
    },
    "formula_002": {
        "reference_answer": (
            "Theo Cockcroft-Gault, nam 65 tuổi, 70 kg, creatinine 1.6 mg/dL có độ thanh thải creatinin khoảng 45.57 ml/ph. "
            "Công thức này dựa trên tuổi, cân nặng, creatinine huyết thanh và giới."
        ),
        "required_facts": [
            "Cockcroft-Gault dùng tuổi, cân nặng, creatinine huyết thanh và giới.",
            "Nam dùng hệ số giới 1.0 trong công thức.",
            "Kết quả creatinine clearance xấp xỉ 45.57 ml/ph.",
        ],
        "relevant_document_ids": [
            "formula::cockcroft_gault",
            "chunk::benh_than_man_p134_002",
            "chunk::benh_than_man_p141_003",
        ],
        "relevant_source_ids": [
            "cockcroft_gault",
            "benh_than_man_p134_002",
            "benh_than_man_p141_003",
        ],
        "eval_tags": ["formula", "cockcroft_gault", "creatinine_clearance", "medical_tool"],
        "eval_notes": "Không nên gọi kết quả này là eGFR chuẩn hóa 1.73m2.",
    },
    "formula_003": {
        "reference_answer": (
            "FENa = (natri niệu x creatinine máu) / (natri máu x creatinine niệu) x 100. "
            "Với UNa 20, PNa 140, UCr 100, PCr 1, FENa khoảng 0.1429%, nhỏ hơn 1%, gợi ý suy thận cấp trước thận trong bối cảnh phù hợp."
        ),
        "required_facts": [
            "Công thức FENa là (urine Na x plasma creatinine)/(plasma Na x urine creatinine) x 100.",
            "Kết quả FENa xấp xỉ 0.1429%.",
            "FENa <1% gợi ý suy thận trước thận.",
        ],
        "relevant_document_ids": [
            "formula::fena_formula",
            "threshold::extra_acute_kidney_injury_fena_lt_1_0_p186_025",
            "chunk::acute_kidney_injury_p186_003",
        ],
        "relevant_source_ids": [
            "fena_formula",
            "extra_acute_kidney_injury_fena_lt_1_0_p186_025",
            "acute_kidney_injury_p186_003",
        ],
        "eval_tags": ["formula", "fena", "vietnamese_aliases", "aki", "medical_tool"],
        "eval_notes": "Cần parse alias tiếng Việt: natri niệu/máu, creatinine niệu/máu.",
    },
    "formula_004": {
        "reference_answer": "Diện tích da cơ thể theo công thức sqrt(cân nặng x chiều cao / 3600). Với 55 kg và 160 cm, BSA khoảng 1.5635 m2.",
        "required_facts": [
            "BSA = sqrt(weight_kg x height_cm / 3600).",
            "Với 55 kg và 160 cm, BSA xấp xỉ 1.5635 m2.",
        ],
        "relevant_document_ids": [
            "formula::body_surface_area",
            "chunk::benh_than_man_p129_003",
        ],
        "relevant_source_ids": [
            "body_surface_area",
            "benh_than_man_p129_003",
        ],
        "eval_tags": ["formula", "bsa", "medical_tool"],
        "eval_notes": "Cần trả về đơn vị m2.",
    },
    "formula_005": {
        "reference_answer": (
            "Với cách viết tiếng Anh, Urine Na 20, plasma Na 140, urine creatinine 100 và plasma creatinine 1 cho FENa khoảng 0.1429%. "
            "Vì nhỏ hơn 1%, kết quả gợi ý suy thận trước thận trong bối cảnh phù hợp."
        ),
        "required_facts": [
            "Urine Na tương ứng urine_na.",
            "Plasma Na tương ứng plasma_na.",
            "Urine creatinine tương ứng urine_creatinine.",
            "Plasma creatinine tương ứng plasma_creatinine.",
            "Kết quả FENa xấp xỉ 0.1429% và <1%.",
        ],
        "relevant_document_ids": [
            "formula::fena_formula",
            "threshold::extra_acute_kidney_injury_fena_lt_1_0_p186_025",
            "chunk::acute_kidney_injury_p186_003",
        ],
        "relevant_source_ids": [
            "fena_formula",
            "extra_acute_kidney_injury_fena_lt_1_0_p186_025",
            "acute_kidney_injury_p186_003",
        ],
        "eval_tags": ["formula", "fena", "english_aliases", "aki", "medical_tool"],
        "eval_notes": "Regression quan trọng: không được tạo false sodium threshold match từ Urine Na/plasma Na.",
    },
    "formula_threshold_001": {
        "reference_answer": (
            "Với nữ 60 tuổi, creatinine 1.4 mg/dL, CKD-EPI 2021 ước tính eGFR khoảng 43.07 ml/ph/1.73m2, thuộc G3b. "
            "ACR 350 mg/g >=300 nên thuộc A3. Kết hợp này cho thấy giảm chức năng thận kèm albumin niệu mức A3; cần diễn giải theo bối cảnh lâm sàng và theo dõi bởi bác sĩ."
        ),
        "required_facts": [
            "CKD-EPI eGFR xấp xỉ 43.07 ml/ph/1.73m2.",
            "43.07 nằm trong khoảng 30 đến dưới 45 nên thuộc G3b.",
            "ACR 350 mg/g >=300 mg/g nên thuộc A3.",
            "Cần nêu cả GFR/eGFR và ACR.",
        ],
        "relevant_document_ids": [
            "formula::ckd_epi_2021_creatinine",
            "threshold::benh_than_man_gfr_between_30_0_p150_002",
            "threshold::benh_than_man_acr_gteq_300_0_p201_003",
            "chunk::benh_than_man_p202_005",
        ],
        "relevant_source_ids": [
            "ckd_epi_2021_creatinine",
            "benh_than_man_gfr_between_30_0_p150_002",
            "benh_than_man_acr_gteq_300_0_p201_003",
            "benh_than_man_p202_005",
        ],
        "eval_tags": ["formula_threshold", "ckd_epi_2021", "gfr", "acr", "g3b", "a3", "medical_tool"],
        "eval_notes": "Case end-to-end quan trọng: route -> tool -> RAG -> final answer.",
    },
    "formula_threshold_002": {
        "reference_answer": (
            "CKD-EPI 2021 cho nữ 60 tuổi, creatinine 1.4 mg/dL ước tính eGFR khoảng 43.07 ml/ph/1.73m2. "
            "Giá trị này thấp hơn 60 và thuộc G3b, nên là bất thường nếu kéo dài/ổn định theo tiêu chuẩn bệnh thận mạn."
        ),
        "required_facts": [
            "eGFR xấp xỉ 43.07 ml/ph/1.73m2.",
            "eGFR 43.07 thuộc G3b.",
            "GFR <60 là bất thường/tiêu chuẩn CKD nếu kéo dài trên 3 tháng.",
            "Không khẳng định bệnh thận mạn chắc chắn nếu thiếu thông tin kéo dài 3 tháng.",
        ],
        "relevant_document_ids": [
            "formula::ckd_epi_2021_creatinine",
            "threshold::benh_than_man_gfr_between_30_0_p150_002",
            "chunk::benh_than_man_p133_002",
        ],
        "relevant_source_ids": [
            "ckd_epi_2021_creatinine",
            "benh_than_man_gfr_between_30_0_p150_002",
            "benh_than_man_p133_002",
        ],
        "eval_tags": ["formula_threshold", "ckd_epi_2021", "gfr", "g3b", "abnormality", "medical_tool"],
        "eval_notes": "Answer nên có caveat về thời gian kéo dài trên 3 tháng.",
    },
    "formula_threshold_003": {
        "reference_answer": (
            "Nam 58 tuổi, creatinine 1.8 mg/dL có CKD-EPI 2021 eGFR khoảng 43.09 ml/ph/1.73m2, thuộc G3b. "
            "ACR 320 mg/g >=300 nên thuộc A3. Câu trả lời cần nêu cả giảm GFR và albumin niệu A3."
        ),
        "required_facts": [
            "CKD-EPI eGFR xấp xỉ 43.09 ml/ph/1.73m2.",
            "43.09 thuộc G3b.",
            "ACR 320 mg/g >=300 mg/g.",
            "ACR >=300 mg/g thuộc A3.",
        ],
        "relevant_document_ids": [
            "formula::ckd_epi_2021_creatinine",
            "threshold::benh_than_man_gfr_between_30_0_p150_002",
            "threshold::benh_than_man_acr_gteq_300_0_p201_003",
            "chunk::benh_than_man_p202_005",
        ],
        "relevant_source_ids": [
            "ckd_epi_2021_creatinine",
            "benh_than_man_gfr_between_30_0_p150_002",
            "benh_than_man_acr_gteq_300_0_p201_003",
            "benh_than_man_p202_005",
        ],
        "eval_tags": ["formula_threshold", "ckd_epi_2021", "gfr", "acr", "g3b", "a3", "medical_tool"],
        "eval_notes": "Case kiểm thử đủ biến tính eGFR nam và phân loại ACR đồng thời.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LangSmith-ready VitalAI eval datasets.")
    parser.add_argument("--skip-test-cases", action="store_true", help="Do not modify tests/cases/*.json.")
    parser.add_argument("--skip-qa-dataset", action="store_true", help="Do not generate data/evaluate_data outputs.")
    parser.add_argument("--top-source-matches", type=int, default=4, help="Max source evidence items per QA case.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_test_cases:
        updated = enrich_test_cases()
        print(f"Enriched {updated} test cases under {CASES_DIR.relative_to(ROOT)}")
    if not args.skip_qa_dataset:
        corpus = load_source_corpus()
        enriched = build_enriched_qa_dataset(corpus=corpus, top_k=args.top_source_matches)
        write_qa_outputs(enriched)
        print(f"Wrote {len(enriched['cases'])} enriched QA cases to {ENRICHED_QA_PATH.relative_to(ROOT)}")
        print(f"Wrote LangSmith JSON to {LANGSMITH_JSON_PATH.relative_to(ROOT)}")
        print(f"Wrote LangSmith JSONL to {LANGSMITH_JSONL_PATH.relative_to(ROOT)}")
    return 0


def enrich_test_cases() -> int:
    updated = 0
    for path in sorted(CASES_DIR.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for case in document.get("cases", []):
            enrichment = CASE_ENRICHMENTS.get(case.get("id"))
            if not enrichment:
                continue
            for key, value in enrichment.items():
                case[key] = value
            updated += 1
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    missing = sorted(set(CASE_ENRICHMENTS) - {case["id"] for path in CASES_DIR.glob("*.json") for case in json.loads(path.read_text(encoding="utf-8")).get("cases", [])})
    if missing:
        raise ValueError(f"Enrichment IDs not found in tests/cases: {missing}")
    return updated


def load_source_corpus() -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    corpus.extend(load_jsonl_docs(PROCESSED_DATA_DIR / "chunks.jsonl", kind="chunk"))
    corpus.extend(load_jsonl_docs(PROCESSED_DATA_DIR / "thresholds.jsonl", kind="threshold"))
    extra_path = PROCESSED_DATA_DIR / "thresholds_extra.jsonl"
    if extra_path.exists():
        corpus.extend(load_jsonl_docs(extra_path, kind="threshold"))
    formulas_path = PROCESSED_DATA_DIR / "formulas.json"
    if formulas_path.exists():
        for item in json.loads(formulas_path.read_text(encoding="utf-8")):
            corpus.append(
                {
                    "document_id": f"formula::{item['formula_id']}",
                    "source_id": item["formula_id"],
                    "source_type": "formula",
                    "content": formula_content(item),
                    "metadata": {
                        "disease_name": item.get("disease_name"),
                        "section_type": item.get("section_type"),
                        "biomarker": None,
                        "source_file": item.get("source_file"),
                        "page": item.get("page"),
                        "language": item.get("language"),
                    },
                }
            )
    for doc in corpus:
        normalized_content = normalize_text(str(doc.get("content") or ""))
        doc["_normalized_content"] = normalized_content
        doc["_tokens"] = token_set(normalized_content)
    return corpus


def load_jsonl_docs(path: Path, *, kind: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if kind == "chunk":
            metadata = item.get("metadata") or {}
            source_id = item["chunk_id"]
            content = item.get("content") or ""
        else:
            metadata = {
                "disease_name": item.get("disease_name"),
                "section_type": item.get("section_type"),
                "biomarker": item.get("biomarker"),
                "source_file": item.get("source_file"),
                "page": item.get("page"),
                "language": item.get("language"),
            }
            source_id = item["threshold_id"]
            content = threshold_content(item)
        docs.append(
            {
                "document_id": f"{kind}::{source_id}",
                "source_id": source_id,
                "source_type": kind,
                "content": content,
                "metadata": metadata,
            }
        )
    return docs


def threshold_content(item: dict[str, Any]) -> str:
    pieces = [
        f"Biomarker: {item.get('biomarker')}",
        f"Rule: {item.get('threshold_op')} {item.get('threshold_value')}",
    ]
    if item.get("threshold_value_min") is not None or item.get("threshold_value_max") is not None:
        pieces.append(f"Range: {item.get('threshold_value_min')} - {item.get('threshold_value_max')}")
    if item.get("threshold_unit"):
        pieces.append(f"Unit: {item.get('threshold_unit')}")
    if item.get("label"):
        pieces.append(f"Label: {item.get('label')}")
    if item.get("source_text"):
        pieces.append(str(item["source_text"]))
    return ". ".join(piece for piece in pieces if piece)


def formula_content(item: dict[str, Any]) -> str:
    variables = ", ".join(str(var.get("name")) for var in item.get("variables", []))
    return (
        f"Công thức {item.get('formula_name')} ({item.get('formula_id')}). "
        f"Biểu thức: {item.get('expression')}. "
        f"Biến đầu vào: {variables}. "
        f"Đầu ra: {item.get('output_name')} ({item.get('output_unit')}). "
        f"Nguồn: {item.get('source_text') or ''}"
    )


def build_enriched_qa_dataset(*, corpus: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    raw_items = json.loads(RAW_QA_PATH.read_text(encoding="utf-8"))
    cases = []
    for index, item in enumerate(raw_items, start=1):
        query = clean_space(item["question"])
        reference_answer = clean_space(item["ground_truth"])
        reference_context = clean_space(item.get("reference_context") or "")
        evidence = match_source_evidence(reference_context or reference_answer, corpus=corpus, top_k=top_k)
        primary_evidence = evidence[:1]
        case_id = f"rag_{index:03d}"
        cases.append(
            {
                "id": case_id,
                "title": query[:100],
                "query": query,
                "top_k": 5,
                "reference_answer": reference_answer,
                "reference_context": reference_context,
                "required_facts": split_required_facts(reference_answer),
                "relevant_document_ids": [doc["document_id"] for doc in primary_evidence],
                "relevant_source_ids": [doc["source_id"] for doc in primary_evidence],
                "source_evidence": evidence,
                "eval_tags": infer_tags(query, reference_answer, evidence),
                "eval_notes": (
                    "Generated from data/evaluate_data/qa_dataset_50_questions.json and matched against "
                    "data/processed_data source artifacts. Review low match_score items before using as hard gold IDs."
                ),
            }
        )
    return {
        "category": "rag_qa",
        "description": "50 QA cases derived from the original renal disease source document, enriched for LangSmith RAG evaluation.",
        "source_dataset": str(RAW_QA_PATH.relative_to(ROOT)),
        "source_artifacts": [
            "data/processed_data/chunks.jsonl",
            "data/processed_data/thresholds.jsonl",
            "data/processed_data/thresholds_extra.jsonl",
            "data/processed_data/formulas.json",
        ],
        "cases": cases,
    }


def match_source_evidence(text: str, *, corpus: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    scored_by_id: dict[str, tuple[float, dict[str, Any]]] = {}
    for part in source_match_parts(text):
        normalized_text = normalize_text(part)
        text_tokens = token_set(normalized_text)
        if not text_tokens:
            continue
        for doc in corpus:
            normalized_content = str(doc.get("_normalized_content") or "")
            if not normalized_content:
                continue
            content_tokens = doc.get("_tokens") or set()
            token_recall = len(text_tokens & content_tokens) / max(1, len(text_tokens))
            exact_bonus = 0.25 if normalized_text and normalized_text[:140] in normalized_content else 0.0
            sequence_score = 0.0
            if token_recall >= 0.12 or exact_bonus:
                sequence_score = SequenceMatcher(None, normalized_text[:1800], normalized_content[:2600]).ratio()
            score = min(1.0, max(token_recall, sequence_score) + exact_bonus)
            if score < 0.18:
                continue
            source_id = str(doc["source_id"])
            current = scored_by_id.get(source_id)
            if current is None or score > current[0]:
                scored_by_id[source_id] = (score, doc)
    scored = list(scored_by_id.values())
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0] if scored else 0.0
    secondary_cutoff = max(0.75, best_score * 0.9)

    evidence = []
    seen: set[str] = set()
    for score, doc in scored:
        if evidence and score < secondary_cutoff:
            continue
        source_id = str(doc["source_id"])
        if source_id in seen:
            continue
        seen.add(source_id)
        metadata = doc.get("metadata") or {}
        evidence.append(
            {
                "document_id": doc["document_id"],
                "source_id": source_id,
                "source_type": doc["source_type"],
                "match_score": round(score, 4),
                "disease_name": metadata.get("disease_name"),
                "section_type": metadata.get("section_type"),
                "biomarker": metadata.get("biomarker"),
                "source_file": metadata.get("source_file"),
                "page": metadata.get("page"),
                "content_preview": clean_space(str(doc.get("content") or ""))[:500],
            }
        )
        if len(evidence) >= top_k:
            break
    return evidence


def source_match_parts(text: str) -> list[str]:
    cleaned = clean_space(text)
    parts = [cleaned]
    for part in re.split(r"\n+|(?<=[.!?。])\s+|;\s+|(?<=:)\s+", str(text or "")):
        part = clean_space(part.strip(" -•0123456789.()"))
        if len(part) >= 18:
            parts.append(part)
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = normalize_text(part)
        if key and key not in seen:
            seen.add(key)
            deduped.append(part)
    return deduped[:12]


def split_required_facts(answer: str) -> list[str]:
    normalized = clean_space(answer)
    parts = re.split(r"(?<=[.!?。])\s+|;\s+|\n+", normalized)
    facts: list[str] = []
    for part in parts:
        part = clean_space(part.strip(" -•"))
        if len(part) < 18:
            continue
        facts.append(part)
    return facts[:8]


def infer_tags(query: str, answer: str, evidence: list[dict[str, Any]]) -> list[str]:
    text = normalize_text(f"{query} {answer}")
    tags = {"rag_qa", "ground_truth"}
    if any(char.isdigit() for char in query + answer):
        tags.add("numeric")
    if "chan doan" in text or "tieu chuan" in text:
        tags.add("diagnosis")
    if "dieu tri" in text or "phac do" in text:
        tags.add("treatment")
    if "tien luong" in text or "tien trien" in text:
        tags.add("prognosis")
    if "phan loai" in text or "giai doan" in text or "class" in text:
        tags.add("classification")
    if any(doc.get("source_type") == "threshold" for doc in evidence):
        tags.add("threshold_evidence")
    if any(doc.get("source_type") == "formula" for doc in evidence):
        tags.add("formula_evidence")
    diseases = {str(doc.get("disease_name")) for doc in evidence if doc.get("disease_name")}
    tags.update(sorted(diseases)[:3])
    return sorted(tags)


def write_qa_outputs(enriched: dict[str, Any]) -> None:
    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ENRICHED_QA_PATH.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    examples = [case_to_langsmith_example(case, enriched["category"], enriched["description"]) for case in enriched["cases"]]
    LANGSMITH_JSON_PATH.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LANGSMITH_JSONL_PATH.write_text(
        "\n".join(json.dumps(example, ensure_ascii=False) for example in examples) + "\n",
        encoding="utf-8",
    )


def case_to_langsmith_example(case: dict[str, Any], category: str, description: str) -> dict[str, Any]:
    example_id = uuid.uuid5(uuid.NAMESPACE_URL, f"vitalai-rag-eval:{category}:{case['id']}")
    return {
        "id": str(example_id),
        "inputs": {
            "query": case["query"],
            "top_k": int(case.get("top_k", 5)),
            "disease_name": case.get("disease_name"),
            "section_type": case.get("section_type"),
            "source_type": case.get("source_type"),
            "biomarker": case.get("biomarker"),
        },
        "outputs": {
            "reference_answer": case.get("reference_answer"),
            "reference_context": case.get("reference_context"),
            "required_facts": case.get("required_facts", []),
            "relevant_document_ids": case.get("relevant_document_ids", []),
            "relevant_source_ids": case.get("relevant_source_ids", []),
            "source_evidence": case.get("source_evidence", []),
        },
        "metadata": {
            "case_id": case["id"],
            "category": category,
            "title": case.get("title", case["id"]),
            "description": description,
            "tags": case.get("eval_tags", []),
            "eval_notes": case.get("eval_notes"),
        },
    }


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text).replace("đ", "d").replace("Đ", "D"))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower()
    value = re.sub(r"[^a-z0-9%/.,<>=+-]+", " ", value)
    value = value.replace(",", ".")
    return clean_space(value)


def token_set(normalized_text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", normalized_text)
        if len(token) > 1 and token not in VI_STOPWORDS
    }


def clean_space(text: str) -> str:
    return " ".join(str(text or "").split())


if __name__ == "__main__":
    raise SystemExit(main())
