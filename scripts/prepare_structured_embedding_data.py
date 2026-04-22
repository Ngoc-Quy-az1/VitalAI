from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def flatten_facts(obj: Any, prefix: str = "") -> list[str]:
    facts: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"language"}:
                continue
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            facts.extend(flatten_facts(value, next_prefix))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            facts.extend(flatten_facts(item, f"{prefix}[{idx}]"))
    else:
        text = str(obj).strip()
        if text:
            facts.append(f"{prefix}: {text}" if prefix else text)
    return facts


def build_embedding_text(doc: dict[str, Any], source_type: str) -> str:
    title = str(doc.get("title") or doc.get("document_id") or "Tài liệu structured")
    disease = str(doc.get("disease") or "không rõ")
    category = str(doc.get("category") or doc.get("graph_type") or "structured_data")
    facts = flatten_facts(doc)
    body = "\n".join(facts[:60])
    return (
        f"Loại nguồn: {source_type}\n"
        f"Tiêu đề: {title}\n"
        f"Bệnh/chủ đề: {disease}\n"
        f"Phân loại: {category}\n"
        "Nội dung có cấu trúc:\n"
        f"{body}"
    )


def convert_file(path: Path, source_type: str, id_prefix: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []

    docs: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        raw_id = str(item.get("document_id") or f"{id_prefix}_{idx+1}")
        document_id = f"{id_prefix}::{raw_id}"
        source_id = raw_id
        content = json.dumps(item, ensure_ascii=False)
        embedding_text = build_embedding_text(item, source_type=source_type)
        docs.append(
            {
                "document_id": document_id,
                "source_type": source_type,
                "source_id": source_id,
                "content": content,
                "embedding_text": embedding_text,
                "metadata": {
                    "doc_type": "structured_metadata",
                    "disease_name": item.get("disease"),
                    "section_type": item.get("category") or item.get("graph_type") or "general",
                    "content_type": "json_structured",
                    "biomarker": None,
                    "source_file": path.name,
                    "page": None,
                    "language": item.get("language", "vi"),
                },
            }
        )
    return docs


def write_jsonl(path: Path, docs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare embedding JSONL from structured metadata files.")
    parser.add_argument(
        "--blocks",
        default="data/processed_data/data_metadata_blocks.json",
        help="Path to data_metadata_blocks.json",
    )
    parser.add_argument(
        "--graphs",
        default="data/processed_data/data_metadata_graphs.json",
        help="Path to data_metadata_graphs.json",
    )
    parser.add_argument(
        "--output",
        default="data/embedding_data/embedding_structured_documents.jsonl",
        help="Output JSONL for embedding/indexing",
    )
    args = parser.parse_args()

    blocks_path = Path(args.blocks)
    graphs_path = Path(args.graphs)
    output_path = Path(args.output)

    docs: list[dict[str, Any]] = []
    if blocks_path.exists():
        docs.extend(convert_file(blocks_path, source_type="structured_table", id_prefix="structured_block"))
    if graphs_path.exists():
        docs.extend(convert_file(graphs_path, source_type="structured_graph", id_prefix="structured_graph"))

    # dedupe by document_id
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in docs:
        did = doc["document_id"]
        if did in seen:
            continue
        seen.add(did)
        deduped.append(doc)

    write_jsonl(output_path, deduped)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output_path),
                "documents": len(deduped),
                "from_blocks": str(blocks_path),
                "from_graphs": str(graphs_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

