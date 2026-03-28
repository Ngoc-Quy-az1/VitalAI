from __future__ import annotations

"""
Điểm chạy CLI cho bước embed và index lên Neon.

Luồng:
1. load `.env`
2. đọc `embedding_documents.jsonl`
3. ensure schema pgvector
4. gọi OpenAI embeddings
5. upsert vào `medical_documents`
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.LLM.embedding.indexer import build_indexer_from_env


async def _run(args: argparse.Namespace) -> dict:
    indexer = build_indexer_from_env(
        input_path=args.input,
        batch_size=args.batch_size,
    )
    return await indexer.run(limit=args.limit)


def main() -> None:
    """Parse CLI args và chạy indexer."""

    parser = argparse.ArgumentParser(description="Embed và upsert document lên Neon pgvector.")
    parser.add_argument(
        "--input",
        default="data/embedding_data/embedding_documents.jsonl",
        help="Đường dẫn tới embedding_documents.jsonl",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Số document mỗi batch gọi embedding API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số document để test nhanh trước khi chạy full.",
    )
    args = parser.parse_args()

    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
