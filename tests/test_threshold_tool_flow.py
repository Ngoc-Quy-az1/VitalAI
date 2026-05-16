from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.medical_tools.service import MedicalToolsService


class ThresholdToolFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = MedicalToolsService()

    def test_acr_classification_survives_ckd_filter(self) -> None:
        result = self.service.evaluate(text="ACR 350 mg/g", disease_name="benh_than_man")
        labels = {(item["threshold"] or {}).get("label") for item in result["classifications"]}

        self.assertIn("A3", labels)

    def test_acr_boundary_300_is_a3(self) -> None:
        result = self.service.evaluate(text="ACR 300 mg/g")
        labels = {(item["threshold"] or {}).get("label") for item in result["classifications"]}

        self.assertIn("A3", labels)
        self.assertNotIn("A2", labels)

    def test_fena_english_aliases_do_not_create_false_sodium_match(self) -> None:
        result = self.service.evaluate(
            text=(
                "Urine Na 20 mmol/L, plasma Na 140 mmol/L, "
                "urine creatinine 100 mg/dL, plasma creatinine 1 mg/dL. Tính FENa."
            ),
            disease_name="acute_kidney_injury",
            formula_ids=["fena_formula"],
        )

        self.assertEqual(result["formula_results"][0]["status"], "computed")
        self.assertAlmostEqual(result["formula_results"][0]["value"], 0.1429, places=4)
        self.assertNotIn("sodium", {item["name"] for item in result["detected_measurements"]})

    def test_hemoglobin_threshold_respects_sex(self) -> None:
        male = self.service.evaluate(text="Nam, Hb 12 g/dL", disease_name="benh_than_man")
        female = self.service.evaluate(text="Nữ, Hb 12 g/dL", disease_name="benh_than_man")

        male_labels = {(item["threshold"] or {}).get("label") for item in male["classifications"]}
        female_labels = {(item["threshold"] or {}).get("label") for item in female["classifications"]}

        self.assertIn("anemia_threshold", male_labels)
        self.assertNotIn("anemia_threshold_female", female_labels)


if __name__ == "__main__":
    unittest.main()
