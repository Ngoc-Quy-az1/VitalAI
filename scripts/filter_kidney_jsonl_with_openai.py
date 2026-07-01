from __future__ import annotations

"""
Lọc JSONL QA bằng OpenAI — tối ưu chi phí:
- Mặc định model: gpt-4o-mini (rẻ, đủ cho phân loại có/không).
- response_format json_object → ít lỗi parse, ít gọi lại.
- max_tokens đầu ra nhỏ (chỉ cần JSON ngắn).
- Cắt answer dài để giảm input tokens.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]


def _read_env_file_var(var_name: str, env_path: Path) -> str:
    if not env_path.exists():
        return ""
    try:
        content = env_path.read_text(encoding="utf-8-sig")
    except OSError:
        return ""
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != var_name:
            continue
        return value.strip().strip("'").strip('"')
    return ""


def _resolve_openai_key() -> str:
    env_value = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_value:
        return env_value
    return _read_env_file_var("OPENAI_API_KEY", ROOT / ".env").strip()


def _extract_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            return None
    brace = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _count_jsonl_records(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


SYSTEM_PROMPT = (
    "Decide if this Q&A is PRIMARILY about kidney/nephrology/urology of the kidneys: "
    "CKD, AKI, GFR, dialysis, kidney transplant, glomerular disease, renal stones as main topic, "
    "kidney function labs, etc. "
    "Return JSON only: is_kidney_related (boolean), confidence (0-1), reason (short English). "
    "Rules — set is_kidney_related to FALSE when: "
    "the main topic is another organ/system (brain/CNS, heart, lung, skin cancer page, etc.) "
    "even if the long answer lists many cancer types including 'kidney cancer' or 'renal cell cancer' in passing; "
    "only a bullet in a metastasis list; a single incidental phrase. "
    "TRUE only when the question or central teaching of the answer is kidney disease or renal care."
)


class OpenAIKidneyClassifier:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_answer_chars: int,
        max_output_tokens: int,
    ) -> None:
        self.client = client
        self.model = model
        self.max_answer_chars = max_answer_chars
        self.max_output_tokens = max_output_tokens

    def classify(self, focus: str, question: str, answer: str) -> tuple[dict | None, str | None]:
        ans = answer if len(answer) <= self.max_answer_chars else answer[: self.max_answer_chars] + "\n...[truncated]"
        user = f"Focus: {focus}\nQuestion: {question}\nAnswer: {ans}"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=self.max_output_tokens,
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = _extract_json_object(raw)
            if not parsed:
                return None, "Không parse được JSON từ model."
            return (
                {
                    "is_kidney_related": bool(parsed.get("is_kidney_related", False)),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "reason": str(parsed.get("reason", "")),
                },
                None,
            )
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"


def smoke_test(client: OpenAI, model: str, max_output_tokens: int) -> None:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Reply with JSON only."},
                {"role": "user", "content": 'Return {"ok": true}'},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=min(64, max_output_tokens),
        )
        _ = r.choices[0].message.content
    except Exception as exc:
        raise SystemExit(f"OpenAI smoke test thất bại: {exc}") from exc


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Lọc JSONL QA bằng OpenAI (tiết kiệm: gpt-4o-mini + ít token).",
    )
    parser.add_argument(
        "--input-jsonl",
        default="data/processed_data/medquad_kidney_qa_keyword.jsonl",
        help="File JSONL đầu vào.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/processed_data/medquad_kidney_qa_openai_filtered.jsonl",
        help="File JSONL đầu ra.",
    )
    parser.add_argument(
        "--summary-json",
        default="data/processed_data/medquad_kidney_qa_openai_filtered_summary.json",
        help="File summary.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="Model (mặc định gpt-4o-mini — rẻ, đủ phân loại).",
    )
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=6000,
        help="Cắt answer để giảm input tokens (mặc định 6000).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=200,
        help="Giới hạn token đầu ra (JSON ngắn; mặc định 200).",
    )
    parser.add_argument(
        "--llm-min-confidence",
        type=float,
        default=0.85,
        help="Ngưỡng confidence để giữ (mặc định 0.85, siết false positive).",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Giới hạn số record (test).")
    parser.add_argument("--progress-every", type=int, default=10, help="Log mỗi N record (0 = tắt).")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Bỏ smoke test.")
    args = parser.parse_args()

    api_key = _resolve_openai_key()
    if not api_key:
        raise SystemExit("Thiếu OPENAI_API_KEY (env hoặc file .env).")

    client = OpenAI(api_key=api_key)

    if not args.skip_smoke_test:
        print("[openai-filter] Smoke test...", flush=True)
        smoke_test(client, args.openai_model, args.max_output_tokens)

    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.exists():
        raise SystemExit(f"Không tìm thấy: {input_jsonl}")

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    print("[openai-filter] Đang đếm dòng...", flush=True)
    line_count = _count_jsonl_records(input_jsonl)
    total_planned = line_count if args.max_records is None else min(line_count, args.max_records)
    print(f"[openai-filter] Dòng: {line_count} | xử lý: {total_planned} | model: {args.openai_model}", flush=True)

    classifier = OpenAIKidneyClassifier(
        client=client,
        model=args.openai_model,
        max_answer_chars=args.max_answer_chars,
        max_output_tokens=args.max_output_tokens,
    )

    total = 0
    kept = 0
    parse_errors = 0
    llm_errors = 0
    first_error_logged = False
    t0 = time.perf_counter()

    with output_jsonl.open("w", encoding="utf-8") as out_f, input_jsonl.open("r", encoding="utf-8") as in_f:
        for line in in_f:
            if not line.strip():
                continue
            if args.max_records is not None and total >= args.max_records:
                break
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            result, err = classifier.classify(
                focus=str(rec.get("focus", "")),
                question=str(rec.get("question", "")),
                answer=str(rec.get("answer", "")),
            )

            if err:
                llm_errors += 1
                if not first_error_logged:
                    print(f"[openai-filter] Lỗi mẫu (record #{total}): {err}", flush=True)
                    first_error_logged = True
                continue

            if not result:
                llm_errors += 1
                continue

            if not result["is_kidney_related"] or float(result["confidence"]) < args.llm_min_confidence:
                continue

            rec["llm_provider"] = "openai"
            rec["llm_model"] = args.openai_model
            rec["llm_reason"] = result["reason"]
            rec["llm_confidence"] = result["confidence"]
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

            if args.progress_every and total % args.progress_every == 0:
                elapsed = time.perf_counter() - t0
                rate = total / elapsed if elapsed > 0 else 0.0
                left = max(0, total_planned - total)
                eta_sec = left / rate if rate > 0 else 0.0
                print(
                    f"[openai-filter] {total}/{total_planned} | giữ {kept} | lỗi parse {parse_errors} | lỗi API {llm_errors} | "
                    f"{elapsed:.1f}s | ~{eta_sec / 60:.1f} phút còn | {rate * 60:.1f} rec/phút",
                    flush=True,
                )

    summary = {
        "input_jsonl": str(input_jsonl.as_posix()),
        "output_jsonl": str(output_jsonl.as_posix()),
        "openai_model": args.openai_model,
        "max_answer_chars": args.max_answer_chars,
        "max_output_tokens": args.max_output_tokens,
        "total_records_processed": total,
        "kept_records": kept,
        "kept_ratio": round((kept / total), 6) if total else 0.0,
        "parse_errors": parse_errors,
        "llm_errors": llm_errors,
        "llm_min_confidence": args.llm_min_confidence,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
