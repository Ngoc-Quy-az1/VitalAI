from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.medical_tools.service import MedicalToolsService
from src.LLM.tools.medical_tools_client import build_structured_context


class FormulaToolFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = MedicalToolsService()

    def test_mdrd_computes_with_default_race_other(self) -> None:
        query = "Nữ 60 tuổi, creatinine 1.4 mg/dL. Hãy ước tính eGFR."
        result = self.service.evaluate(text=query, formula_ids=["mdrd_gfr"])

        formula = result["formula_results"][0]
        self.assertEqual(formula["formula_id"], "mdrd_gfr")
        self.assertEqual(formula["status"], "computed")
        self.assertAlmostEqual(formula["value"], 40.7681, places=4)
        self.assertIn("assumptions", formula)
        self.assertTrue(any("race=other" in item for item in formula["assumptions"]))

        structured_context = build_structured_context(result, query=query)
        self.assertIn("40.7681", structured_context)
        self.assertIn("race=other", structured_context)

    def test_formula_missing_inputs_appear_in_structured_context(self) -> None:
        query = "Tính Cockcroft-Gault với creatinine 1.6 mg/dL."
        result = self.service.evaluate(text=query, formula_ids=["cockcroft_gault"])
        structured_context = build_structured_context(result, query=query)

        self.assertEqual(result["formula_results"][0]["status"], "missing_inputs")
        self.assertIn("thiếu", structured_context)


if __name__ == "__main__":
    unittest.main()
