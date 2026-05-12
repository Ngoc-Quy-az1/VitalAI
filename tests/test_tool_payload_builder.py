from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.LLM.tools.medical_tools.request_builder import build_tool_input_payload, sanitize_tool_parameters
from src.LLM.tools.medical_tools.router_plan import normalize_router_plan


def contains_null(value):
    if value is None:
        return True
    if isinstance(value, dict):
        return any(contains_null(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_null(item) for item in value)
    return False


class ToolPayloadBuilderTests(unittest.TestCase):
    def test_build_tool_input_payload_extracts_supported_measurements_without_nulls(self) -> None:
        payload = build_tool_input_payload("Nữ 60 tuổi, creatinine 1.4 mg/dL, ACR 350 mg/g. Hãy tính eGFR.")

        self.assertEqual(payload["text"], "Nữ 60 tuổi, creatinine 1.4 mg/dL, ACR 350 mg/g. Hãy tính eGFR.")
        self.assertIn("measurements", payload)
        self.assertEqual(payload["measurements"]["sex"]["value"], "female")
        self.assertEqual(payload["measurements"]["age"]["value"], 60.0)
        self.assertNotIn("unit", payload["measurements"]["age"])
        self.assertEqual(payload["measurements"]["creatinine"]["unit"], "mg/dL")
        self.assertEqual(payload["measurements"]["creatinine_mg_dl"]["value"], 1.4)
        self.assertEqual(payload["measurements"]["ACR"]["value"], 350.0)
        self.assertFalse(contains_null(payload))

    def test_sanitize_tool_parameters_removes_nulls_and_unknown_fields(self) -> None:
        extracted_payload = build_tool_input_payload("Nam 65 tuổi, nặng 70 kg, creatinine 1.6 mg/dL. Tính Cockcroft-Gault.")
        normalized = normalize_router_plan(
            {
                "needs_medical_tool": True,
                "tool_call": {
                    "endpoint": "/mcp/medical-tools/evaluate",
                    "parameters": {
                        "text": None,
                        "measurements": {
                            "unknown_field": {"value": 123},
                            "weight_kg": {"value": 70, "unit": "kg"},
                        },
                        "disease_name": None,
                        "formula_ids": ["cockcroft_gault", "fake_formula"],
                        "include_debug": True,
                    },
                },
                "rag_plan": {"should_retrieve": True, "query": "x", "filters": {}},
            },
            extracted_payload["text"],
            extracted_payload=extracted_payload,
        )

        params = normalized["tool_call"]["parameters"]
        self.assertEqual(params["text"], extracted_payload["text"])
        self.assertEqual(params["formula_ids"], ["cockcroft_gault"])
        self.assertFalse(params["include_debug"])
        self.assertIn("measurements", params)
        self.assertNotIn("unknown_field", params["measurements"])
        self.assertNotIn("disease_name", params)

    def test_sanitize_tool_parameters_keeps_non_null_minimal_shape(self) -> None:
        params = sanitize_tool_parameters(
            {"text": "ACR 350 mg/g", "measurements": None, "formula_ids": None, "disease_name": None},
            query="ACR 350 mg/g",
            extracted_payload={"text": "ACR 350 mg/g"},
        )
        self.assertEqual(
            params,
            {
                "text": "ACR 350 mg/g",
                "formula_ids": [],
                "include_debug": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
