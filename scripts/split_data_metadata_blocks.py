from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed_data" / "data_metadata_blocks.json"
TABLES_PATH = ROOT / "data" / "processed_data" / "data_metadata_tables.json"
GRAPHS_PATH = ROOT / "data" / "processed_data" / "data_metadata_graphs.json"


def is_graph_doc(item: dict) -> bool:
    return bool(
        item.get("graph_type")
        or (isinstance(item.get("nodes"), list) and isinstance(item.get("edges"), list))
        or item.get("root_node")
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy input: {INPUT_PATH}")

    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input blocks phải là JSON array.")

    tables: list[dict] = []
    graphs: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if is_graph_doc(item):
            graphs.append(item)
        else:
            tables.append(item)

    TABLES_PATH.write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")
    GRAPHS_PATH.write_text(json.dumps(graphs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "input_path": str(INPUT_PATH),
                "tables_path": str(TABLES_PATH),
                "graphs_path": str(GRAPHS_PATH),
                "tables_count": len(tables),
                "graphs_count": len(graphs),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
