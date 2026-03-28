from __future__ import annotations

"""
Script QA cho các artifact đã extract.

File này tồn tại vì đọc thẳng JSONL bằng mắt rất chậm và dễ sót lỗi.
Script sẽ tạo một report để trả lời nhanh:
- đã extract được bao nhiêu chunk / threshold / formula
- còn thiếu metadata bao nhiêu
- có threshold nào nhìn là biết sai rõ ràng theo một vài rule sanity-check hay không

Đây không phải bộ kiểm định y khoa. Nó chỉ là lớp QA cấu trúc cho phase ingestion.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Đọc một file JSONL UTF-8 vào bộ nhớ."""

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def build_report(base: Path) -> dict:
    """
    Tạo QA report từ các output hiện tại trong thư mục processed.

    Các check ở đây được giữ đơn giản và dễ giải thích.
    Chỉ flag những trường hợp rất có khả năng là lỗi extraction.
    """

    chunks = load_jsonl(base / "chunks.jsonl")
    thresholds = load_jsonl(base / "thresholds.jsonl")
    formulas = json.loads((base / "formulas.json").read_text(encoding="utf-8"))

    suspicious_thresholds = []
    for item in thresholds:
        if item["biomarker"] == "cholesterol" and item["threshold_unit"] == "g/L":
            suspicious_thresholds.append(item)
        if item["biomarker"] == "GFR" and item["threshold_unit"] not in ("ml/ph/1.73m2", "ml/ph", "ml/phút"):
            suspicious_thresholds.append(item)
        if item["biomarker"] == "ACR" and item["threshold_unit"] not in ("mg/g", "mg/mmol", None):
            suspicious_thresholds.append(item)
        if item["biomarker"] == "PCR" and item["threshold_unit"] not in ("mg/g", "mg/mmol", None):
            suspicious_thresholds.append(item)
        if item["biomarker"] == "creatinine" and item["threshold_unit"] in ("mg/g", "mg/mmol"):
            suspicious_thresholds.append(item)

    report = {
        "counts": {
            "chunks": len(chunks),
            "thresholds": len(thresholds),
            "formulas": len(formulas),
        },
        "missing_metadata": {
            "chunks_without_disease_name": sum(1 for item in chunks if item["metadata"]["disease_name"] is None),
            "thresholds_without_disease_name": sum(1 for item in thresholds if item["disease_name"] is None),
            "formulas_without_disease_name": sum(1 for item in formulas if item["disease_name"] is None),
        },
        "distributions": {
            "chunk_doc_type": dict(Counter(item["metadata"]["doc_type"] for item in chunks)),
            "chunk_section_type": dict(Counter(item["metadata"]["section_type"] for item in chunks)),
            "threshold_biomarker": dict(Counter(item["biomarker"] for item in thresholds)),
            "formula_name": dict(Counter(item["formula_name"] for item in formulas)),
        },
        "samples": {
            "unknown_chunks": [item for item in chunks if item["metadata"]["disease_name"] is None][:5],
            "unknown_thresholds": [item for item in thresholds if item["disease_name"] is None][:10],
            "suspicious_thresholds": suspicious_thresholds[:10],
            "formulas": formulas,
        },
    }
    return report


def main() -> None:
    """Ghi file QA report JSON cạnh các artifact đã xử lí."""

    parser = argparse.ArgumentParser(description="Tạo QA report cho các output dữ liệu đã xử lí.")
    parser.add_argument(
        "--input",
        default="data/processed_data",
        help="Thư mục chứa chunks.jsonl, thresholds.jsonl và formulas.json.",
    )
    parser.add_argument(
        "--output",
        default="data/processed_data/qa_report.json",
        help="Đường dẫn file JSON sẽ được ghi QA report.",
    )
    args = parser.parse_args()

    base = Path(args.input)
    output = Path(args.output)
    report = build_report(base)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"qa_report": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
