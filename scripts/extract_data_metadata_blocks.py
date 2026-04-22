from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed_data" / "chunks.jsonl"
OUTPUT_PATH = ROOT / "data" / "processed_data" / "data_metadata_blocks.json"


def load_full_text_from_chunks(path: Path) -> str:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        content = str(row.get("content") or "")
        if content:
            lines.append(content)
    return "\n\n".join(lines)


def _extract_balanced_json(text: str, start_idx: int) -> str | None:
    depth = 0
    in_string = False
    escape = False
    begin = -1
    for i in range(start_idx, len(text)):
        ch = text[i]
        if begin < 0:
            if ch == "{":
                begin = i
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[begin : i + 1]
    return None


def _repair_json_text(s: str) -> str:
    out = s.replace("\ufeff", "")
    out = re.sub(r",\s*([}\]])", r"\1", out)
    out = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", out)
    return out.strip()


def _try_parse_candidate(candidate: str) -> dict[str, Any] | None:
    repaired = _repair_json_text(candidate)
    try:
        obj = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and obj.get("document_id"):
        return obj
    return None


def extract_blocks(full_text: str) -> list[dict[str, Any]]:
    marker = "Data metadata:"
    starts = [m.start() for m in re.finditer(re.escape(marker), full_text)]
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for idx, marker_pos in enumerate(starts):
        seg_end = starts[idx + 1] if idx + 1 < len(starts) else min(len(full_text), marker_pos + 50000)
        segment = full_text[marker_pos:seg_end]

        candidate_offsets: list[int] = []
        for m in re.finditer(r'"document_id"\s*:', segment):
            back = segment.rfind("{", 0, m.start())
            if back >= 0:
                candidate_offsets.append(back)
        first_curly = segment.find("{")
        if first_curly >= 0:
            candidate_offsets.append(first_curly)

        offsets: list[int] = []
        seen_offset: set[int] = set()
        for off in candidate_offsets:
            if off not in seen_offset:
                seen_offset.add(off)
                offsets.append(off)

        picked: dict[str, Any] | None = None
        for off in offsets:
            raw_obj = _extract_balanced_json(segment, off)
            if not raw_obj:
                continue
            obj = _try_parse_candidate(raw_obj)
            if obj:
                picked = obj
                break

        if not picked:
            continue
        key = f"{picked.get('document_id')}::{picked.get('title')}"
        if key in seen:
            continue
        seen.add(key)
        results.append(picked)

    return results


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file chunks: {CHUNKS_PATH}")

    full_text = load_full_text_from_chunks(CHUNKS_PATH)
    blocks = extract_blocks(full_text)
    OUTPUT_PATH.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "chunks_path": str(CHUNKS_PATH),
                "output_path": str(OUTPUT_PATH),
                "metadata_blocks_found": len(blocks),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
