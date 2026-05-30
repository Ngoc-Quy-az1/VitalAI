from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.LLM.qa.graph import (
    _assess_evidence_quality,
    _build_agentic_subqueries,
    _build_agentic_retry_query,
    _grade_and_sort_evidence_items,
    _should_agentic_retry,
)


class AgenticRagGraphTests(unittest.TestCase):
    def test_evidence_judge_requests_retry_for_weak_context(self) -> None:
        state = {
            "route": "retrieve",
            "query": "Hội chứng thận hư chẩn đoán theo tiêu chuẩn nào?",
            "agentic_retry_count": 0,
            "evidence_items": [
                {
                    "content": "Đây là đoạn nói về xét nghiệm máu chung và không liên quan đến câu hỏi.",
                    "fusion_score": 0.02,
                }
            ],
            "retrieval_plan": {
                "soft_hints": {
                    "terms": ["protein niệu", "albumin máu"],
                    "disease_names": ["hoi_chung_than_hu"],
                    "section_types": ["diagnosis_criteria"],
                    "biomarkers": ["protein_niệu_24h"],
                }
            },
        }

        quality = _assess_evidence_quality(state)

        self.assertFalse(quality["sufficient"])
        self.assertTrue(_should_agentic_retry(state, quality))

    def test_agentic_retry_query_uses_soft_hints(self) -> None:
        state = {
            "query": "Hội chứng thận hư chẩn đoán theo tiêu chuẩn nào?",
            "retrieval_plan": {
                "soft_hints": {
                    "terms": ["protein niệu", "albumin máu"],
                    "disease_names": ["hoi_chung_than_hu"],
                    "section_types": ["diagnosis_criteria"],
                    "biomarkers": ["protein_niệu_24h"],
                }
            },
        }

        retry_query = _build_agentic_retry_query(state)

        self.assertIn("Hội chứng thận hư", retry_query)
        self.assertIn("Chẩn đoán", retry_query)
        self.assertIn("protein niệu", retry_query)
        self.assertIn("protein_niệu_24h", retry_query)

    def test_agentic_subqueries_decompose_soft_hints(self) -> None:
        plan = {
            "soft_hints": {
                "terms": ["ARA 1997", "protein niệu"],
                "disease_names": ["lupus_nephritis"],
                "section_types": ["diagnosis_criteria"],
                "biomarkers": ["protein_niệu_24h"],
            }
        }

        subqueries = _build_agentic_subqueries(
            "Viêm cầu thận lupus được chẩn đoán khi nào theo ARA 1997?",
            plan,
        )

        joined = "\n".join(subqueries)
        self.assertGreaterEqual(len(subqueries), 2)
        self.assertIn("Viêm thận lupus", joined)
        self.assertIn("Chẩn đoán", joined)
        self.assertIn("protein_niệu_24h", joined)

    def test_evidence_grading_promotes_more_relevant_chunk(self) -> None:
        state = {
            "query": "Hội chứng thận hư chẩn đoán theo tiêu chuẩn protein niệu albumin máu?",
            "filters": {},
            "retrieval_plan": {
                "soft_hints": {
                    "terms": ["protein niệu", "albumin máu"],
                    "disease_names": ["hoi_chung_than_hu"],
                    "section_types": ["diagnosis_criteria"],
                    "biomarkers": ["protein_niệu_24h"],
                }
            },
        }
        evidence = [
            {
                "document_id": "weak",
                "content": "Đoạn nói về xét nghiệm máu chung.",
                "fusion_score": 0.03,
            },
            {
                "document_id": "strong",
                "content": "Hội chứng thận hư chẩn đoán với protein niệu cao và albumin máu giảm.",
                "fusion_score": 0.02,
            },
        ]

        sorted_items, grades = _grade_and_sort_evidence_items(evidence, state, limit=2)

        self.assertEqual(sorted_items[0]["document_id"], "strong")
        self.assertEqual(grades[0]["document_id"], "strong")
        self.assertGreater(grades[0]["score"], grades[1]["score"])


if __name__ == "__main__":
    unittest.main()
