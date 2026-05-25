from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.LLM.retrieval.query_planner import build_retrieval_plan
from src.LLM.retrieval.vector_search import NeonVectorSearcher


class RetrievalQueryPlannerTests(unittest.TestCase):
    def test_lupus_ara_question_stays_soft_filtered_without_tool_result(self) -> None:
        plan = build_retrieval_plan(
            query="Viêm cầu thận lupus được chẩn đoán khi nào theo tiêu chuẩn ARA 1997?",
            initial_filters={},
            router_plan={},
            extracted_tool_payload={},
            medical_tool_result=None,
        )

        self.assertIsNone(plan["filters"]["disease_name"])
        self.assertIn("lupus_nephritis", plan["soft_hints"]["disease_names"])
        self.assertIn("ara 1997", plan["soft_hints"]["terms"])
        self.assertIn("Viêm thận lupus", plan["query"])

    def test_general_glomerular_question_stays_soft_filtered(self) -> None:
        plan = build_retrieval_plan(
            query="Bệnh cầu thận có những biểu hiện chung nào?",
            initial_filters={},
            router_plan={},
            extracted_tool_payload={},
            medical_tool_result=None,
        )

        self.assertIsNone(plan["filters"]["disease_name"])
        self.assertIn("benh_ly_cau_than", plan["soft_hints"]["disease_names"])

    def test_tool_result_can_harden_ckd_threshold_filter(self) -> None:
        tool_result = {
            "threshold_matches": [
                {
                    "matched": True,
                    "biomarker": "ACR",
                    "threshold": {
                        "disease_name": "benh_than_man",
                        "label": "A3",
                    },
                    "source": {"source_text": "ACR >= 300 mg/g thuộc A3"},
                }
            ]
        }
        plan = build_retrieval_plan(
            query="ACR 350 mg/g có ý nghĩa gì?",
            initial_filters={},
            router_plan={"needs_medical_tool": True},
            extracted_tool_payload={"measurements": {"ACR": {"value": 350, "unit": "mg/g"}}},
            medical_tool_result=tool_result,
        )

        self.assertEqual(plan["query_type"], "threshold")
        self.assertEqual(plan["filters"]["disease_name"], "benh_than_man")
        self.assertIn("ACR", plan["soft_hints"]["biomarkers"])
        self.assertIn("ACR >= 300 mg/g thuộc A3", plan["query"])

    def test_searcher_uses_primary_line_for_keyword_query(self) -> None:
        searcher = object.__new__(NeonVectorSearcher)
        understanding = searcher._understand_query(
            query=(
                "Bệnh cầu thận thay đổi tối thiểu thường gặp ở đối tượng nào?\n"
                "Bệnh trọng tâm: Bệnh lý cầu thận\n"
                "Mục cần tìm: Lâm sàng và cận lâm sàng"
            ),
            disease_name=None,
            section_type=None,
            biomarker=None,
        )

        self.assertEqual(
            understanding["keyword_query"],
            "Bệnh cầu thận thay đổi tối thiểu thường gặp ở đối tượng nào?",
        )
        self.assertIn("thay doi toi thieu", understanding["key_phrases"])

    def test_searcher_merges_short_parent_heading_into_context(self) -> None:
        searcher = object.__new__(NeonVectorSearcher)
        row = {
            "document_id": "chunk::benh_ly_cau_than_p3_005",
            "chunk_index": 5,
            "content": "4.1.1. Lâm sàng và cận lâm sàng\nThường gặp ở trẻ nhỏ tuổi đi học.",
        }
        neighbors = [
            {
                "document_id": "chunk::benh_ly_cau_than_p3_004",
                "chunk_index": 4,
                "content": "4.1. Bệnh cầu thận thay đổi tối thiểu: hội chứng thận hư tiên phát",
                "section_type": "general",
            },
            row,
        ]

        merged = searcher._merge_neighbor_context(row, neighbors)

        self.assertIn("Bệnh cầu thận thay đổi tối thiểu", merged)
        self.assertIn("Thường gặp ở trẻ nhỏ tuổi đi học", merged)

    def test_searcher_does_not_merge_previous_non_heading_into_heading_context(self) -> None:
        searcher = object.__new__(NeonVectorSearcher)
        row = {
            "document_id": "chunk::hoi_chung_than_hu_p20_004",
            "chunk_index": 4,
            "content": "5. CHẨN ĐOÁN XÁC ĐỊNH Tiêu chuẩn chẩn đoán hội chứng thận hư bao gồm:",
        }
        neighbors = [
            {
                "document_id": "chunk::hoi_chung_than_hu_p20_003",
                "chunk_index": 3,
                "content": "Công thức máu có chỉ số hồng cầu, hemoglobin, hematocrit thường giảm.",
                "section_type": "general",
            },
            row,
            {
                "document_id": "chunk::hoi_chung_than_hu_p20_005",
                "chunk_index": 5,
                "content": "Protein niệu > 3,5g/24 giờ/1,73m diện tích bề mặt cơ thể.",
                "section_type": "general",
            },
        ]

        merged = searcher._merge_neighbor_context(row, neighbors)

        self.assertNotIn("Công thức máu", merged)
        self.assertIn("Protein niệu > 3,5g/24 giờ", merged)


if __name__ == "__main__":
    unittest.main()
