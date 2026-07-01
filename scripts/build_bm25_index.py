import json
import os
import sys
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH để import đúng các module
sys.path.append(str(Path(__file__).parent.parent))

# Cấu hình UTF-8 cho stdout trên Windows console để tránh UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.LLM.retrieval.bm25_retriever import BM25IndexBuilder, default_bm25_index_path

def build_real_bm25_index():
    # Thư mục gốc dự án
    root_dir = Path(__file__).resolve().parent.parent
    
    # Các file chứa chunks y khoa thực tế sau giai đoạn Prepare
    input_paths = [
        root_dir / "data" / "embedding_data" / "embedding_documents.jsonl",
        root_dir / "data" / "embedding_data" / "embedding_structured_documents.jsonl"
    ]
    
    chunks = []
    for path in input_paths:
        if path.exists():
            print(f"-> Đang đọc dữ liệu từ: {path}")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
        else:
            print(f"-> Không tìm thấy file: {path} (Bỏ qua)")
            
    if not chunks:
        print("❌ LỖI: Không tìm thấy bất kỳ chunk dữ liệu thật nào để đánh chỉ mục!")
        print("Vui lòng chạy các bước trích xuất và chuẩn bị dữ liệu (process_medical_data.py & prepare_embedding_data.py) trước.")
        sys.exit(1)
        
    output_path = default_bm25_index_path()
    print(f"-> Đang xây dựng chỉ mục BM25 cho {len(chunks)} chunks dữ liệu thật...")
    
    # Thực hiện build chỉ mục và lưu thành file .pkl
    BM25IndexBuilder.build_and_save(chunks, output_path)
    print(f"✅ THÀNH CÔNG: Đã lưu BM25 Index thật vào: {output_path}")

if __name__ == "__main__":
    build_real_bm25_index()
