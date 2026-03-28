from __future__ import annotations

"""
Script CLI để test retrieval trực tiếp trên Neon.

Cách dùng cơ bản:
- truyền một query
- nhận top-k kết quả gần nhất theo vector similarity
- có thể filter theo disease / section / source_type / biomarker
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.LLM.retrieval.vector_search import build_searcher_from_env


async def _run(args: argparse.Namespace) -> dict:
    searcher = build_searcher_from_env()
    return await searcher.search(
        query=args.query,
        top_k=args.top_k,
        disease_name=args.disease_name,
        section_type=args.section_type,
        source_type=args.source_type,
        biomarker=args.biomarker,
    )


def main() -> None:
    """Parse CLI args và in kết quả retrieval dưới dạng JSON."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Test semantic retrieval trên Neon pgvector.")
    parser.add_argument("--query", required=True, help="Câu hỏi hoặc truy vấn cần test.")
    parser.add_argument("--top-k", type=int, default=5, help="Số kết quả trả về.")
    parser.add_argument("--disease-name", default=None, help="Filter metadata theo disease_name.")
    parser.add_argument("--section-type", default=None, help="Filter metadata theo section_type.")
    parser.add_argument("--source-type", default=None, help="Filter theo source_type: chunk|threshold|formula.")
    parser.add_argument("--biomarker", default=None, help="Filter metadata theo biomarker.")
    args = parser.parse_args()

    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
