from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz


def resolve_pdf_path(source: Path) -> Path:
    if source.is_file():
        return source
    pdf_files = sorted(source.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Không tìm thấy file PDF trong: {source}")
    if len(pdf_files) > 1:
        raise ValueError(f"Tìm thấy nhiều PDF trong {source}, hãy chỉ định --source cụ thể.")
    return pdf_files[0]


def load_full_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    try:
        return "\n".join(doc[i].get_text("text") for i in range(doc.page_count))
    finally:
        doc.close()


def extract_balanced_json(text: str, start_pos: int) -> tuple[str | None, int]:
    in_string = False
    escaped = False
    depth = 0
    begin = -1
    for i in range(start_pos, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                begin = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and begin >= 0:
                    return text[begin : i + 1], i + 1
    return None, start_pos


def repair_json_text(raw: str) -> str:
    # normalize linebreak/tab inside strings and remove trailing commas
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in raw:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = not in_string
            continue
        if in_string and ch in {"\n", "\r", "\t"}:
            out.append(" ")
            continue
        out.append(ch)
    repaired = "".join(out).replace("\ufeff", "")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired.strip()


def extract_blocks(full_text: str, marker: str) -> list[str]:
    blocks: list[str] = []
    marker_regex = re.compile(re.escape(marker).replace(r"\ ", r"\s+"), flags=re.IGNORECASE)
    pos = 0
    while True:
        found = marker_regex.search(full_text, pos)
        if not found:
            break
        json_start = full_text.find("{", found.end())
        if json_start < 0:
            pos = found.end()
            continue
        block, end_pos = extract_balanced_json(full_text, json_start)
        if block:
            blocks.append(block)
            pos = end_pos
        else:
            pos = found.end()
    return blocks


def dedupe_by_document_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        did = str(item.get("document_id") or "").strip()
        title = str(item.get("title") or "").strip()
        key = f"{did}::{title}"
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Extract Data metadata blocks from PDF.")
    parser.add_argument("--source", required=True, help="PDF path hoặc thư mục chứa PDF.")
    parser.add_argument(
        "--output",
        default="data/processed_data/data_metadata_blocks.json",
        help="Output JSON chứa metadata parse thành công.",
    )
    parser.add_argument(
        "--error-output",
        default="data/processed_data/data_metadata_parse_errors.json",
        help="Output JSON chứa block parse lỗi để audit.",
    )
    parser.add_argument("--marker", default="Data metadata:", help="Marker bắt đầu JSON.")
    args = parser.parse_args()

    pdf_path = resolve_pdf_path(Path(args.source))
    full_text = load_full_text(pdf_path)
    raw_blocks = extract_blocks(full_text, marker=args.marker)

    parsed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for idx, block_text in enumerate(raw_blocks, start=1):
        repaired = repair_json_text(block_text)
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict) and obj.get("document_id"):
                parsed.append(obj)
            else:
                errors.append({"index": idx, "error": "JSON không phải object có document_id", "snippet": repaired[:500]})
        except json.JSONDecodeError as exc:
            errors.append({"index": idx, "error": str(exc), "snippet": repaired[:500]})

    parsed = dedupe_by_document_id(parsed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    err_path = Path(args.error_output)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "source": str(pdf_path),
                "marker": args.marker,
                "blocks_found": len(raw_blocks),
                "parsed_ok": len(parsed),
                "parse_errors": len(errors),
                "output": str(output_path),
                "error_output": str(err_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

