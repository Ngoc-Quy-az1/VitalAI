from __future__ import annotations

"""Export chatbot answers and run inputs/outputs from LangSmith experiments.

This script connects to LangSmith, retrieves runs for a specific experiment (project),
and exports the queries, generated answers, routes, and metadata to a JSON file.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langsmith import Client

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export chatbot answers from LangSmith runs.")
    parser.add_argument(
        "--project",
        help="Name of the LangSmith project/experiment. If not specified, it will list available projects.",
    )
    parser.add_argument(
        "--output",
        help="Path to save the exported answers (JSON format).",
    )
    return parser.parse_args()

def extract_answer(outputs: Any) -> str:
    if not outputs:
        return ""
    if isinstance(outputs, str):
        return outputs.strip()
    if isinstance(outputs, dict):
        for key in ("answer", "output", "content", "text", "response", "result"):
            val = outputs.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            elif isinstance(val, dict):
                content = val.get("content") or val.get("text")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""

def extract_contexts(outputs: Any) -> list[str]:
    contexts: list[str] = []
    if not outputs or not isinstance(outputs, dict):
        return contexts
    for key in ("contexts", "rag_contexts", "tool_contexts"):
        val = outputs.get(key)
        if isinstance(val, list):
            contexts.extend(str(item) for item in val if str(item or "").strip())
    for key in ("context", "rag_context", "tool_context"):
        val = outputs.get(key)
        if isinstance(val, str) and val.strip():
            contexts.append(val.strip())
    return list(dict.fromkeys(contexts)) # deduplicate preserving order


def main() -> None:
    load_dotenv()
    args = parse_args()

    print("Đang kết nối tới LangSmith...")
    client = Client()

    # Determine project name
    project_name = args.project
    if not project_name:
        # Fallback to LANGSMITH_PROJECT env var
        project_name = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
        
    # If still not found, list available projects
    if not project_name:
        try:
            projects = list(client.list_projects())
            print("\nKhông tìm thấy project cụ thể. Danh sách các project có sẵn trên LangSmith của bạn:")
            for p in projects:
                print(f" - {p.name} (ID: {p.id})")
            print("\nVui lòng chạy lại script kèm tham số --project <tên-project>.")
            sys.exit(1)
        except Exception as exc:
            print(f"Lỗi khi liệt kê các project từ LangSmith: {exc}")
            sys.exit(1)

    print(f"Đang tìm thông tin project: '{project_name}'")
    try:
        # Verify the project exists
        project = client.read_project(project_name=project_name)
        print(f"Tìm thấy project ID: {project.id}")
    except Exception as exc:
        print(f"Lỗi: Không thể tìm thấy hoặc truy cập project '{project_name}' trên LangSmith: {exc}")
        # Try to suggest similar projects
        try:
            projects = list(client.list_projects())
            print("\nCác project có sẵn trên LangSmith của bạn:")
            for p in projects:
                print(f" - {p.name}")
        except Exception:
            pass
        sys.exit(1)

    output_path = args.output
    if not output_path:
        # Create a clean safe name for the output file
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_name)
        output_path = str(ROOT / "tests" / "results" / f"langsmith_answers_{safe_name}.json")

    print("Đang tải danh sách runs...")
    runs = list(
        client.list_runs(
            project_name=project_name,
            is_root=True,
            select=["id", "inputs", "outputs", "reference_example_id", "error", "start_time", "end_time", "feedback_stats"],
        )
    )
    print(f"Đã tải {len(runs)} root runs.")

    if not runs:
        print("Không có runs nào được tìm thấy trong experiment này.")
        sys.exit(0)

    # If there are reference examples, we can map them to obtain references
    # Fetching examples mapping to avoid querying one by one
    print("Đang lấy thông tin liên kết từ dataset (nếu có)...")
    dataset_examples = {}
    try:
        dataset_id = project.reference_dataset_id
        if dataset_id:
            examples = list(client.list_examples(dataset_id=dataset_id))
            for ex in examples:
                dataset_examples[str(ex.id)] = ex
            print(f"Đã nạp {len(dataset_examples)} examples từ dataset tham chiếu.")
    except Exception as exc:
        print(f"Không thể tải dataset chi tiết: {exc}. Tiếp tục trích xuất run data.")

    exported_runs = []
    for run in runs:
        # Extract inputs
        run_inputs = run.inputs or {}
        query = run_inputs.get("query") or run_inputs.get("input") or ""

        # Extract outputs
        run_outputs = run.outputs or {}
        answer = extract_answer(run_outputs)
        route = run_outputs.get("route")
        
        # Gather contexts retrieved
        contexts = extract_contexts(run_outputs)

        # Map reference info if available
        ref_example = None
        reference_answer = None
        reference_context = None
        required_facts = []
        if run.reference_example_id and str(run.reference_example_id) in dataset_examples:
            ref_example = dataset_examples[str(run.reference_example_id)]
            ref_outputs = ref_example.outputs or {}
            reference_answer = ref_outputs.get("reference_answer")
            reference_context = ref_outputs.get("reference_context")
            required_facts = ref_outputs.get("required_facts") or []

        # Feedback stats
        feedbacks = {}
        if hasattr(run, "feedback_stats") and run.feedback_stats:
            for k, v in run.feedback_stats.items():
                if isinstance(v, dict):
                    feedbacks[k] = v.get("avg")

        exported_runs.append({
            "run_id": str(run.id),
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "query": query,
            "answer": answer,
            "route": route,
            "contexts_retrieved": contexts,
            "reference": {
                "example_id": str(run.reference_example_id) if run.reference_example_id else None,
                "reference_answer": reference_answer,
                "reference_context": reference_context,
                "required_facts": required_facts,
            },
            "feedbacks": feedbacks,
            "error": run.error,
        })

    # Save to file
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(exported_runs, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"\nThành công! Đã xuất dữ liệu trả lời vào file: {out_file.absolute()}")

if __name__ == "__main__":
    main()
