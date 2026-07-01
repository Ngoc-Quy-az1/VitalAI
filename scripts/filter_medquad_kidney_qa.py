from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KIDNEY_TERMS = {
    "kidney",
    "renal",
    "nephro",
    "nephritis",
    "nephropathy",
    "nephrotic",
    "nephritic",
    "ckd",
    "aki",
    "glomerul",
    "gfr",
    "egfr",
    "creatinine",
    "albuminuria",
    "proteinuria",
    "dialysis",
    "hemodialysis",
    "peritoneal dialysis",
    "kidney stone",
    "urolithiasis",
    "hydronephrosis",
    "hematuria",
    "polycystic kidney",
    "kidney failure",
    "chronic kidney disease",
    "acute kidney injury",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _looks_like_kidney(text: str, focus: str) -> bool:
    merged = f"{focus} {text}".lower()
    if "adrenal" in merged:
        return False
    return any(term in merged for term in KIDNEY_TERMS)


def _iter_xml_files(folder: Path, limit: int | None) -> Iterable[Path]:
    count = 0
    for path in sorted(folder.rglob("*.xml")):
        yield path
        count += 1
        if limit is not None and count >= limit:
            return


def _extract_records(xml_path: Path) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    source = root.attrib.get("source", "")
    url = root.attrib.get("url", "")
    focus = _normalize_text(root.findtext("Focus", default=""))

    out: list[dict] = []
    for qa in root.findall("./QAPairs/QAPair"):
        question_node = qa.find("Question")
        answer_node = qa.find("Answer")
        question = _normalize_text(question_node.text if question_node is not None else "")
        answer = _normalize_text(answer_node.text if answer_node is not None else "")
        if not question or not answer:
            continue
        out.append(
            {
                "xml_file": str(xml_path.as_posix()),
                "document_id": root.attrib.get("id", ""),
                "source": source,
                "url": url,
                "focus": focus,
                "qtype": (question_node.attrib.get("qtype", "") if question_node is not None else ""),
                "question": question,
                "answer": answer,
            }
        )
    return out


def _count_xml_files(folder: Path) -> int:
    return sum(1 for _ in folder.rglob("*.xml"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Lọc MedQuAD XML theo từ khóa để lấy QA liên quan bệnh thận (không dùng LLM).",
    )
    parser.add_argument(
        "--input-dir",
        default="data/MedQuAD-master",
        help="Thư mục gốc chứa dataset MedQuAD (các file .xml).",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/processed_data/medquad_kidney_qa.jsonl",
        help="File JSONL đầu ra.",
    )
    parser.add_argument(
        "--summary-json",
        default="data/processed_data/medquad_kidney_qa_summary.json",
        help="File summary thống kê.",
    )
    parser.add_argument(
        "--limit-xml",
        type=int,
        default=None,
        help="Giới hạn số file xml (test nhanh).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="In tiến trình mỗi N file xml (0 = tắt).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Không tìm thấy thư mục input: {input_dir}")

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    xml_total_target = _count_xml_files(input_dir)
    if args.limit_xml is not None:
        xml_total_target = min(xml_total_target, args.limit_xml)

    xml_total = 0
    qa_pairs_raw = 0
    qa_pairs_with_answer = 0
    qa_kidney = 0
    parse_errors = 0
    source_counter: dict[str, int] = {}
    t0 = time.perf_counter()

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for xml_path in _iter_xml_files(input_dir, args.limit_xml):
            xml_total += 1
            try:
                root = ET.parse(xml_path).getroot()
                qa_pairs_raw += len(root.findall("./QAPairs/QAPair"))
                records = _extract_records(xml_path)
            except Exception:
                parse_errors += 1
                continue
            qa_pairs_with_answer += len(records)
            for rec in records:
                context_text = f"{rec['question']} {rec['answer']}"
                if not _looks_like_kidney(context_text, rec["focus"]):
                    continue
                qa_kidney += 1
                source = rec.get("source", "unknown") or "unknown"
                source_counter[source] = source_counter.get(source, 0) + 1
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if args.progress_every and xml_total % args.progress_every == 0:
                elapsed = time.perf_counter() - t0
                rate = xml_total / elapsed if elapsed > 0 else 0.0
                remaining = max(0, xml_total_target - xml_total)
                eta_sec = remaining / rate if rate > 0 else 0.0
                print(
                    f"[keyword] XML {xml_total}/{xml_total_target} | QA giữ: {qa_kidney} | "
                    f"{elapsed:.1f}s | ~{eta_sec / 60:.1f} phút còn lại",
                    flush=True,
                )

    summary = {
        "input_dir": str(input_dir.as_posix()),
        "output_jsonl": str(output_jsonl.as_posix()),
        "xml_files_scanned": xml_total,
        "xml_parse_errors": parse_errors,
        "qa_pairs_raw": qa_pairs_raw,
        "qa_pairs_with_non_empty_answer": qa_pairs_with_answer,
        "qa_pairs_kidney": qa_kidney,
        "kidney_ratio_over_non_empty": round((qa_kidney / qa_pairs_with_answer), 6) if qa_pairs_with_answer else 0.0,
        "source_distribution": dict(sorted(source_counter.items(), key=lambda kv: kv[1], reverse=True)),
        "keywords_used": sorted(KIDNEY_TERMS),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
