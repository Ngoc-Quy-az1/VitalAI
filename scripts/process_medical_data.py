from __future__ import annotations

"""
Điểm chạy CLI cho bước xử lí data đầu tiên.

Dùng script này khi muốn tạo lại các artifact có cấu trúc từ file PDF thô.

Script này không làm các việc sau:
- không embed dữ liệu
- không ghi vào database
- không xây agent

Nó chỉ chuyển file PDF hiện tại thành các file cục bộ trong `data/processed_data/`
để có thể kiểm tra chất lượng extraction trước khi đi sang bước tiếp theo.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.LLM.ingestion.processor import MedicalPdfProcessor


def main() -> None:
    """Xác định file PDF nguồn, chạy processor, rồi in ra các đường dẫn output đã ghi."""

    parser = argparse.ArgumentParser(description="Xử lí PDF thận học thành các output có cấu trúc.")
    parser.add_argument(
        "--source",
        default="data/raw_data",
        help="Đường dẫn tới file PDF hoặc thư mục chỉ chứa một file PDF nguồn.",
    )
    parser.add_argument(
        "--output",
        default="data/processed_data",
        help="Thư mục sẽ ghi ra chunks.jsonl, thresholds.jsonl, formulas.json và summary.json.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if source.is_file():
        pdf_path = source
    else:
        pdf_files = sorted(source.glob("*.pdf"))
        if not pdf_files:
            raise SystemExit(f"Không tìm thấy file PDF nào trong: {source}")
        if len(pdf_files) > 1:
            raise SystemExit(f"Giai đoạn này chỉ mong đợi một file PDF nguồn, nhưng tìm thấy {len(pdf_files)} file.")
        pdf_path = pdf_files[0]

    processor = MedicalPdfProcessor(pdf_path)
    output_paths = processor.write_outputs(args.output)
    print(json.dumps({key: str(value) for key, value in output_paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
