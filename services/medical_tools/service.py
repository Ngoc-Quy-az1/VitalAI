from __future__ import annotations

"""Structured threshold/formula service deploy độc lập với AI service."""

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.medical_tools.aliases import (
    BIOMARKER_ALIASES,
    FORMULA_OUTPUT_TO_BIOMARKER,
    FORMULA_VARIABLE_ALIASES,
    UNIT_ALIASES,
)
from services.medical_tools.safe_eval import FormulaEvaluationError, expression_names, safe_eval_expression


NUMBER_RE = r"-?\d+(?:[.,]\d+)?"
UNIT_RE = r"g/24\s*(?:giờ|gio|h)|g/l|g/dl|mmol/l|mg/g|mg/mmol|mg/dl|μmol/l|µmol/l|umol/l|ml/ph(?:út|ut)?/1[.,]73m2|ml/ph(?:út|ut)?|mmhg|kg|cm|m2|%"


@dataclass(frozen=True)
class ParsedValue:
    name: str
    value: float
    unit: str | None = None
    source: str = "input"


class MedicalToolsService:
    """Engine đọc `thresholds.jsonl` + `formulas.json` và trả structured result."""

    def __init__(self, processed_data_dir: str | Path = "data/processed_data") -> None:
        self.processed_data_dir = Path(processed_data_dir)
        self.thresholds = self._load_all_thresholds()
        self.formulas = self._load_formulas(self.processed_data_dir / "formulas.json")
        self.formula_by_id = {item["formula_id"]: item for item in self.formulas}
        self.structured_tables = self._load_structured_array(
            preferred=self.processed_data_dir / "data_metadata_tables.json",
            fallback=self.processed_data_dir / "data_metadata_blocks.json",
        )
        self.structured_graphs = self._load_structured_array(
            preferred=self.processed_data_dir / "data_metadata_graphs.json",
            fallback=None,
        )

    def capabilities(self) -> dict[str, Any]:
        """Metadata cho MCP/AI service biết tool này hỗ trợ gì."""

        return {
            "service": "vitalai-medical-tools",
            "version": "v1",
            "threshold_biomarkers": sorted({item["biomarker"] for item in self.thresholds}),
            "formulas": [
                {
                    "formula_id": item["formula_id"],
                    "formula_name": item.get("formula_name"),
                    "output_name": item.get("output_name"),
                    "output_unit": item.get("output_unit"),
                    "variables": [self._normalize_variable_name(v.get("name", "")) for v in item.get("variables", [])],
                }
                for item in self.formulas
            ],
            "structured_table_ids": [item.get("document_id") for item in self.structured_tables if item.get("document_id")],
            "structured_graph_ids": [item.get("document_id") for item in self.structured_graphs if item.get("document_id")],
        }

    def graph_query(self, *, query: str, document_id: str | None = None, top_k: int = 3) -> dict[str, Any]:
        candidates = self.structured_graphs
        if document_id:
            candidates = [item for item in candidates if item.get("document_id") == document_id]

        ranked = self._rank_structured_documents(query=query, documents=candidates, doc_type="structured_graph", top_k=top_k)
        return {
            "result_type": "structured_graph_query",
            "query": query,
            "document_id": document_id,
            "hits": ranked,
        }

    def query_structured_knowledge(self, *, query: str, top_k: int = 5) -> dict[str, Any]:
        table_hits = self._rank_structured_documents(
            query=query,
            documents=self.structured_tables,
            doc_type="structured_table",
            top_k=max(top_k, 3),
        )
        graph_hits = self._rank_structured_documents(
            query=query,
            documents=self.structured_graphs,
            doc_type="structured_graph",
            top_k=max(top_k, 3),
        )
        merged_map: dict[str, dict[str, Any]] = {}
        for hit in [*table_hits, *graph_hits]:
            doc_id = str(hit.get("document_id") or "")
            if not doc_id:
                continue
            current = merged_map.get(doc_id)
            if current is None or float(hit.get("score") or 0.0) > float(current.get("score") or 0.0):
                merged_map[doc_id] = hit
        merged = sorted(merged_map.values(), key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]
        return {
            "result_type": "structured_knowledge_query",
            "query": query,
            "hits": merged,
            "tables_found": len(table_hits),
            "graphs_found": len(graph_hits),
        }

    def evaluate(
        self,
        *,
        text: str | None = None,
        measurements: Any = None,
        disease_name: str | None = None,
        formula_ids: list[str] | None = None,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        """Parse input, tính công thức nếu đủ biến, rồi so threshold/class."""

        parsed = self._parse_text(text or "")
        explicit = self._normalize_measurements(measurements)

        measurement_values = {**parsed["measurements"], **explicit["measurements"]}
        formula_values = {**parsed["formula_variables"], **explicit["formula_variables"]}
        categorical_values = {**parsed["categorical"], **explicit["categorical"]}

        formula_results = self._evaluate_formulas(
            formula_values=formula_values,
            categorical_values=categorical_values,
            formula_ids=formula_ids,
        )

        derived_measurements: dict[str, ParsedValue] = {}
        for result in formula_results:
            if result["status"] != "computed":
                continue
            biomarker = FORMULA_OUTPUT_TO_BIOMARKER.get(result.get("output_name"))
            if not biomarker:
                continue
            derived_measurements[biomarker] = ParsedValue(
                name=biomarker,
                value=result["value"],
                unit=result.get("unit"),
                source=f"formula:{result['formula_id']}",
            )

        # Chỉ số người dùng nhập trực tiếp được ưu tiên khi trùng tên với chỉ số tính ra.
        # Ví dụ user có sẵn GFR và cũng đủ biến tính MDRD eGFR: threshold nên đánh giá GFR họ cung cấp,
        # còn MDRD vẫn được trả riêng trong formula_results/derived_measurements.
        all_measurements = {**derived_measurements, **measurement_values}
        threshold_evaluations = self._evaluate_thresholds(
            measurements=all_measurements,
            disease_name=disease_name,
        )
        threshold_evaluations = self._dedupe_evaluations(threshold_evaluations)
        threshold_matches = [item for item in threshold_evaluations if item["matched"]]
        classifications = [item for item in threshold_matches if item["threshold"].get("label")]

        response = {
            "input": {
                "text": text,
                "disease_name": disease_name,
            },
            "detected_measurements": [self._serialize_value(value) for value in measurement_values.values()],
            "derived_measurements": [self._serialize_value(value) for value in derived_measurements.values()],
            "threshold_matches": threshold_matches,
            "threshold_evaluations": threshold_evaluations,
            "classifications": classifications,
            "formula_results": formula_results,
            "safety": {
                "medical_disclaimer": "Kết quả chỉ mang tính tham khảo, không thay thế đánh giá của bác sĩ.",
                "unit_warning": "Chỉ so sánh trực tiếp khi đơn vị input khớp hoặc không có đơn vị trong dữ liệu ngưỡng.",
            },
        }
        if include_debug:
            response["debug"] = {
                "formula_variables": {key: self._serialize_value(value) for key, value in formula_values.items()},
                "categorical": categorical_values,
                "threshold_count": len(self.thresholds),
                "formula_count": len(self.formulas),
            }
        return response

    def _evaluate_formulas(
        self,
        *,
        formula_values: dict[str, ParsedValue],
        categorical_values: dict[str, str],
        formula_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        selected = [self.formula_by_id[item] for item in formula_ids or [] if item in self.formula_by_id]
        if formula_ids is None:
            selected = self.formulas

        results: list[dict[str, Any]] = []
        for formula in selected:
            expression = formula["expression"]
            names = expression_names(expression)
            variables: dict[str, float] = {}
            missing: list[str] = []

            if "sex_factor" in names:
                sex = self._normalize_sex(categorical_values.get("sex"))
                if sex is None:
                    missing.append("sex")
                else:
                    variables["sex_factor"] = self._sex_factor(formula["formula_id"], sex)

            if "race_factor" in names:
                race = self._normalize_race(categorical_values.get("race"))
                if race is None:
                    missing.append("race")
                else:
                    variables["race_factor"] = 1.21 if race == "black" else 1.0

            for name in sorted(names - {"sex_factor", "race_factor"}):
                if name not in formula_values:
                    missing.append(name)
                else:
                    variables[name] = formula_values[name].value

            base = {
                "formula_id": formula["formula_id"],
                "formula_name": formula.get("formula_name"),
                "output_name": formula.get("output_name"),
                "unit": formula.get("output_unit"),
                "required_variables": sorted(names),
            }
            if missing:
                results.append({**base, "status": "missing_inputs", "missing_inputs": sorted(set(missing))})
                continue

            try:
                value = safe_eval_expression(expression, variables)
            except (FormulaEvaluationError, ZeroDivisionError, OverflowError) as exc:
                results.append({**base, "status": "error", "error": str(exc)})
                continue

            results.append({**base, "status": "computed", "value": round(value, 4)})
        return results

    def _evaluate_thresholds(
        self,
        *,
        measurements: dict[str, ParsedValue],
        disease_name: str | None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for biomarker, parsed in measurements.items():
            candidates = [item for item in self.thresholds if item["biomarker"] == biomarker]
            if disease_name:
                disease_matches = [item for item in candidates if item.get("disease_name") == disease_name]
                if disease_matches:
                    candidates = disease_matches

            for threshold in candidates:
                comparison_value = self._convert_value(parsed.value, parsed.unit, threshold.get("threshold_unit"))
                if comparison_value is None:
                    continue
                matched = self._matches_threshold(comparison_value, threshold)

                unit = threshold.get("threshold_unit")
                unit_matches = self._unit_matches(parsed.unit, unit)
                results.append(
                    {
                        "biomarker": biomarker,
                        "input_value": parsed.value,
                        "input_unit": parsed.unit,
                        "comparison_value": comparison_value,
                        "comparison_unit": unit or parsed.unit,
                        "matched": matched,
                        "status": "condition_met" if matched else "condition_not_met",
                        "unit_matches": unit_matches,
                        "threshold": self._serialize_threshold(threshold),
                        "source": self._safe_source(threshold),
                    }
                )
        return results

    def _parse_text(self, text: str) -> dict[str, Any]:
        measurements: dict[str, ParsedValue] = {}
        formula_variables: dict[str, ParsedValue] = {}
        categorical: dict[str, str] = {}

        normalized = self._normalize_ascii(text)
        sex = self._detect_sex(normalized)
        race = self._detect_race(normalized)
        if sex:
            categorical["sex"] = sex
        if race:
            categorical["race"] = race

        for name, aliases in BIOMARKER_ALIASES.items():
            parsed = self._extract_value_for_aliases(text, aliases)
            if parsed:
                measurements[name] = ParsedValue(name=name, value=parsed[0], unit=parsed[1], source="text")

        for name, aliases in FORMULA_VARIABLE_ALIASES.items():
            if name in {"sex", "race"}:
                continue
            parsed = self._extract_value_for_aliases(text, aliases)
            if parsed:
                formula_variables[name] = ParsedValue(name=name, value=parsed[0], unit=parsed[1], source="text")

        # Creatinine trong text thường vừa là biomarker, vừa là input mg/dL cho công thức.
        if "creatinine" in measurements and "creatinine_mg_dl" not in formula_variables:
            item = measurements["creatinine"]
            if item.unit in {None, "mg/dL"}:
                formula_variables["creatinine_mg_dl"] = ParsedValue(
                    name="creatinine_mg_dl",
                    value=item.value,
                    unit="mg/dL",
                    source=item.source,
                )

        # Creatinine máu là input tương đương cho MDRD/Cockcroft khi đơn vị là mg/dL.
        if "plasma_creatinine" in formula_variables and "creatinine_mg_dl" not in formula_variables:
            item = formula_variables["plasma_creatinine"]
            if item.unit in {None, "mg/dL"}:
                formula_variables["creatinine_mg_dl"] = ParsedValue(
                    name="creatinine_mg_dl",
                    value=item.value,
                    unit="mg/dL",
                    source=item.source,
                )

        if "LDL_cholesterol" in measurements and "cholesterol" in measurements:
            ldl = measurements["LDL_cholesterol"]
            cholesterol = measurements["cholesterol"]
            if ldl.value == cholesterol.value and ldl.unit == cholesterol.unit and "ldl" in normalized:
                del measurements["cholesterol"]

        bp = re.search(r"(?P<systolic>\d{2,3})\s*/\s*(?P<diastolic>\d{2,3})\s*(?:mmHg|mmhg)?", text)
        if bp and ("huyết áp" in text.lower() or "ha" in normalized or "blood pressure" in normalized):
            measurements["systolic_bp"] = ParsedValue("systolic_bp", self._to_float(bp.group("systolic")) or 0.0, "mmHg", "text")
            measurements["diastolic_bp"] = ParsedValue("diastolic_bp", self._to_float(bp.group("diastolic")) or 0.0, "mmHg", "text")

        return {"measurements": measurements, "formula_variables": formula_variables, "categorical": categorical}

    def _normalize_measurements(self, measurements: Any) -> dict[str, Any]:
        result = {"measurements": {}, "formula_variables": {}, "categorical": {}}
        if not measurements:
            return result

        items: list[dict[str, Any]] = []
        if isinstance(measurements, dict):
            for key, value in measurements.items():
                if isinstance(value, dict):
                    items.append({"name": key, **value})
                else:
                    items.append({"name": key, "value": value})
        elif isinstance(measurements, list):
            items = [item for item in measurements if isinstance(item, dict)]

        for item in items:
            raw_name = str(item.get("name") or item.get("biomarker") or item.get("variable") or "").strip()
            if not raw_name:
                continue

            if raw_name in {"sex", "race"}:
                if item.get("value") is not None:
                    result["categorical"][raw_name] = str(item["value"])
                continue

            canonical_biomarker = self._canonical_name(raw_name, BIOMARKER_ALIASES)
            canonical_variable = self._canonical_name(raw_name, FORMULA_VARIABLE_ALIASES)
            value = self._to_float(item.get("value"))
            if value is None:
                continue
            unit = self._normalize_unit(item.get("unit"))

            if canonical_biomarker:
                result["measurements"][canonical_biomarker] = ParsedValue(
                    name=canonical_biomarker,
                    value=value,
                    unit=unit,
                    source="explicit",
                )
            if canonical_variable:
                result["formula_variables"][canonical_variable] = ParsedValue(
                    name=canonical_variable,
                    value=value,
                    unit=unit,
                    source="explicit",
                )

        if "creatinine" in result["measurements"] and "creatinine_mg_dl" not in result["formula_variables"]:
            item = result["measurements"]["creatinine"]
            if item.unit in {None, "mg/dL"}:
                result["formula_variables"]["creatinine_mg_dl"] = ParsedValue(
                    name="creatinine_mg_dl",
                    value=item.value,
                    unit="mg/dL",
                    source=item.source,
                )
        if "plasma_creatinine" in result["formula_variables"] and "creatinine_mg_dl" not in result["formula_variables"]:
            item = result["formula_variables"]["plasma_creatinine"]
            if item.unit in {None, "mg/dL"}:
                result["formula_variables"]["creatinine_mg_dl"] = ParsedValue(
                    name="creatinine_mg_dl",
                    value=item.value,
                    unit="mg/dL",
                    source=item.source,
                )
        return result

    def _extract_value_for_aliases(self, text: str, aliases: list[str]) -> tuple[float, str | None] | None:
        for alias in sorted(aliases, key=len, reverse=True):
            alias_pattern = re.escape(alias).replace("\\ ", r"\s+")
            patterns = [
                rf"(?i)(?<!\w){alias_pattern}(?!\w)\s*(?:=|:|là|la)?\s*(?P<value>{NUMBER_RE})\s*(?P<unit>{UNIT_RE})?",
                rf"(?i)(?P<value>{NUMBER_RE})\s*(?P<unit>{UNIT_RE})?\s*(?<!\w){alias_pattern}(?!\w)",
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if not match:
                    continue
                value = self._to_float(match.group("value"))
                if value is None:
                    continue
                return value, self._normalize_unit(match.groupdict().get("unit"))
        return None

    def _matches_threshold(self, value: float, threshold: dict[str, Any]) -> bool:
        op = threshold.get("threshold_op")
        target = threshold.get("threshold_value")
        if op == ">":
            return value > float(target)
        if op == ">=":
            return value >= float(target)
        if op == "<":
            return value < float(target)
        if op == "<=":
            return value <= float(target)
        if op == "between":
            min_value = float(threshold.get("threshold_value_min", target))
            max_value = float(threshold.get("threshold_value_max", target))
            upper_ok = value <= max_value if self._between_upper_inclusive(threshold) else value < max_value
            return min_value <= value and upper_ok
        return False

    def _serialize_threshold(self, threshold: dict[str, Any]) -> dict[str, Any]:
        return {
            "threshold_id": threshold.get("threshold_id"),
            "op": threshold.get("threshold_op"),
            "value": threshold.get("threshold_value"),
            "value_min": threshold.get("threshold_value_min"),
            "value_max": threshold.get("threshold_value_max"),
            "unit": threshold.get("threshold_unit"),
            "label": threshold.get("label"),
            "severity": threshold.get("severity"),
            "disease_name": threshold.get("disease_name"),
            "section_type": threshold.get("section_type"),
        }

    def _safe_source(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_file": item.get("source_file"),
            "source_text": item.get("source_text"),
            "disease_name": item.get("disease_name"),
            "section_type": item.get("section_type"),
        }

    def _serialize_value(self, value: ParsedValue) -> dict[str, Any]:
        return {"name": value.name, "value": value.value, "unit": value.unit, "source": value.source}

    def _unit_matches(self, input_unit: str | None, threshold_unit: str | None) -> bool | None:
        if input_unit is None or threshold_unit is None:
            return None
        return self._normalize_unit(input_unit) == self._normalize_unit(threshold_unit)

    def _convert_value(self, value: float, input_unit: str | None, threshold_unit: str | None) -> float | None:
        normalized_input = self._normalize_unit(input_unit)
        normalized_threshold = self._normalize_unit(threshold_unit)
        if normalized_input is None or normalized_threshold is None or normalized_input == normalized_threshold:
            return value
        if normalized_input == "g/dL" and normalized_threshold == "g/L":
            return value * 10
        if normalized_input == "g/L" and normalized_threshold == "g/dL":
            return value / 10
        return None

    def _between_upper_inclusive(self, threshold: dict[str, Any]) -> bool:
        source = str(threshold.get("source_text") or "")
        max_value = threshold.get("threshold_value_max")
        if max_value is None:
            return False
        max_text = str(float(max_value)).rstrip("0").rstrip(".")
        return re.search(rf"<=\s*{re.escape(max_text)}(?:\D|$)", source) is not None

    def _canonical_name(self, raw_name: str, aliases: dict[str, list[str]]) -> str | None:
        normalized = self._normalize_ascii(raw_name)
        for canonical, values in aliases.items():
            candidate_aliases = [canonical, *values]
            if any(self._normalize_ascii(alias) == normalized for alias in candidate_aliases):
                return canonical
        return None

    def _normalize_variable_name(self, name: str) -> str:
        return name.split("(", 1)[0].strip()

    def _normalize_unit(self, unit: Any) -> str | None:
        if unit is None:
            return None
        value = str(unit).strip()
        if not value:
            return None
        normalized = value.replace(" ", "")
        return UNIT_ALIASES.get(value, UNIT_ALIASES.get(normalized, value))

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip().replace(",", "."))
        except ValueError:
            return None

    def _detect_sex(self, normalized_text: str) -> str | None:
        if re.search(r"(?<![a-z0-9])(nu|female|woman)(?![a-z0-9])", normalized_text):
            return "female"
        if re.search(r"(?<![a-z0-9])(nam|male|man)(?![a-z0-9])", normalized_text):
            return "male"
        return None

    def _detect_race(self, normalized_text: str) -> str | None:
        if re.search(r"(?<![a-z0-9])(black|da den|african)(?![a-z0-9])", normalized_text):
            return "black"
        if re.search(r"(?<![a-z0-9])(other|khac|chau a|asian)(?![a-z0-9])", normalized_text):
            return "other"
        return None

    def _normalize_sex(self, value: str | None) -> str | None:
        if value is None:
            return None
        return self._detect_sex(self._normalize_ascii(value))

    def _normalize_race(self, value: str | None) -> str | None:
        if value is None:
            return None
        return self._detect_race(self._normalize_ascii(value)) or "other"

    def _sex_factor(self, formula_id: str, sex: str) -> float:
        if formula_id == "cockcroft_gault":
            return 0.85 if sex == "female" else 1.0
        if formula_id == "mdrd_gfr":
            return 0.742 if sex == "female" else 1.0
        return 1.0

    def _normalize_ascii(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
        return " ".join(ascii_text.split())

    def _load_thresholds(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy thresholds file: {path}")
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _load_all_thresholds(self) -> list[dict[str, Any]]:
        thresholds = self._load_thresholds(self.processed_data_dir / "thresholds.jsonl")
        extra_path = self.processed_data_dir / "thresholds_extra.jsonl"
        if extra_path.exists():
            thresholds.extend(self._load_thresholds(extra_path))
        return self._dedupe_thresholds(thresholds)

    def _dedupe_thresholds(self, thresholds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for item in thresholds:
            key = (
                item.get("biomarker"),
                item.get("threshold_op"),
                item.get("threshold_value"),
                item.get("threshold_value_min"),
                item.get("threshold_value_max"),
                item.get("threshold_unit"),
                item.get("disease_name"),
                item.get("section_type"),
                item.get("page"),
                str(item.get("source_text") or "")[:120],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _dedupe_evaluations(self, evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for item in evaluations:
            threshold = item["threshold"]
            key = (
                item.get("biomarker"),
                item.get("matched"),
                threshold.get("op"),
                threshold.get("value"),
                threshold.get("value_min"),
                threshold.get("value_max"),
                threshold.get("unit"),
                threshold.get("label"),
                threshold.get("disease_name"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _load_formulas(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy formulas file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_structured_array(self, *, preferred: Path, fallback: Path | None) -> list[dict[str, Any]]:
        source = preferred if preferred.exists() else fallback
        if source is None or not source.exists():
            return []
        data = json.loads(source.read_text(encoding="utf-8"))
        return [item for item in data if isinstance(item, dict)]

    def _rank_structured_documents(
        self,
        *,
        query: str,
        documents: list[dict[str, Any]],
        doc_type: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_tokens = self._tokenize(self._normalize_ascii(query))
        if not query_tokens:
            return []

        ranked: list[dict[str, Any]] = []
        for item in documents:
            text = self._flatten_json(item)
            normalized = self._normalize_ascii(text)
            tokens = self._tokenize(normalized)
            if not tokens:
                continue
            overlap = len(query_tokens.intersection(tokens))
            if overlap <= 0:
                continue
            score = overlap / max(len(query_tokens), 1)
            score += self._structured_query_boost(query=self._normalize_ascii(query), normalized_document=normalized)
            ranked.append(
                {
                    "document_id": item.get("document_id"),
                    "source_type": doc_type,
                    "score": round(score, 4),
                    "preview": text[:500],
                    "content": item,
                }
            )

        ranked.sort(key=lambda value: value["score"], reverse=True)
        return ranked[:top_k]

    def _flatten_json(self, value: Any) -> str:
        chunks: list[str] = []
        self._collect_json_text(value, chunks)
        return " | ".join(part for part in chunks if part)

    def _collect_json_text(self, value: Any, chunks: list[str]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    self._collect_json_text(item, chunks)
                else:
                    chunks.append(f"{key}: {item}")
            return
        if isinstance(value, list):
            for item in value:
                self._collect_json_text(item, chunks)
            return
        if value is not None:
            chunks.append(str(value))

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9_]{2,}", text) if token}

    def _structured_query_boost(self, *, query: str, normalized_document: str) -> float:
        score = 0.0
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(normalized_document)

        # Exact phrase boosts for strongly identifiable table/graph families.
        phrase_pairs = (
            ("rifle", "rifle"),
            ("kdigo", "kdigo"),
            ("r risk", "risk"),
            ("injury", "injury"),
            ("failure", "failure"),
            ("loss", "loss"),
            ("end stage", "end-stage"),
            ("acute kidney injury", "acute kidney injury"),
            ("suy than cap", "suy than cap"),
        )
        for q_phrase, d_phrase in phrase_pairs:
            if q_phrase in query and d_phrase in normalized_document:
                score += 0.25

        # Disambiguate "R-risk" questions toward RIFLE instead of CKD risk maps.
        if ("r-risk" in query or "r risk" in query) and "rifle" in normalized_document:
            score += 0.9
        if ("r-risk" in query or "r risk" in query) and "classification_system: rifle" in normalized_document:
            score += 0.6

        # Boost when uncommon medical stage tokens align.
        special_tokens = {"rifle", "risk", "injury", "failure", "loss", "aki", "kdigo"}
        matched_special = special_tokens.intersection(query_tokens).intersection(doc_tokens)
        if matched_special:
            score += 0.12 * len(matched_special)

        return score
