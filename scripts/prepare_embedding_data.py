from __future__ import annotations

"""
Điểm chạy CLI cho bước chuẩn bị embedding.

Script này dùng sau khi đã có:
- `chunks.jsonl`
- `thresholds.jsonl`
- `formulas.json`

Nó không gọi model embedding.
Nó chỉ tạo ra một bộ document thống nhất, gọn hơn và phù hợp hơn
để sau này batch-embed lên vector database.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.LLM.embedding.preparation import EmbeddingDataPreparer


def main() -> None:
    """Đọc processed data và ghi ra embedding-ready artifacts."""

    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu embedding từ các output đã extract.")
    parser.add_argument(
        "--input",
        default="data/processed_data",
        help="Thư mục chứa chunks.jsonl, thresholds.jsonl và formulas.json.",
    )
    parser.add_argument(
        "--output",
        default="data/embedding_data",
        help="Thư mục sẽ ghi embedding_documents.jsonl và embedding_manifest.json.",
    )
    args = parser.parse_args()

    preparer = EmbeddingDataPreparer(args.input)
    output_paths = preparer.write_outputs(args.output)
    print(json.dumps({key: str(value) for key, value in output_paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
