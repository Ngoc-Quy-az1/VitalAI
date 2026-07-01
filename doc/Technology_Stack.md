# Danh Mục Công Nghệ Áp Dụng (Technology Stack)

Tài liệu này tổng hợp chi tiết toàn bộ các công nghệ, thư viện, framework và mô hình trí tuệ nhân tạo (AI) đang được áp dụng trong mã nguồn của dự án **VitalAI**.

---

## 1. Ngôn Ngữ & Kiến Trúc Backend Core

- **Python (>= 3.10)**: Ngôn ngữ lập trình chính cho toàn bộ backend xử lý logic AI, trích xuất dữ liệu, dịch vụ công cụ y học và kiểm thử.
- **FastAPI (0.115.7)**: Framework web bất đồng bộ (Asynchronous) hiệu năng cao được sử dụng để xây dựng hai dịch vụ API chính:
  - **AI Service** (`http://localhost:8008`): Phục vụ các endpoint chat `/chat/answer` (đồng bộ) và `/chat/stream` (truyền dữ liệu dạng Server-Sent Events - SSE).
  - **Medical Tools Service** (`http://localhost:8010`): Dịch vụ chuyên biệt tính toán công thức y khoa và đối chiếu ngưỡng chỉ số.
- **Uvicorn (0.34.0)**: ASGI server phân phối ứng dụng FastAPI đảm bảo xử lý concurrency tốt.

---

## 2. Công Cụ Điều Phối Agent & Quản Lý Luồng (Agent Orchestration)

- **LangGraph (0.2.76)**: Framework cốt lõi của LangChain dùng để điều phối luồng logic Agentic RAG. Nó cho phép xây dựng một máy trạng thái (State Graph) có tính bất đồng bộ cao, cho phép phân nhánh có điều kiện, rollback/retry khi dữ liệu truy xuất yếu, và duy trì ngữ cảnh trạng thái trong quá trình xử lý câu hỏi y khoa.
- **LangChain / LangChain Core (0.3.83)**: Cung cấp lớp trừu tượng hóa (abstraction layer) để giao tiếp đồng bộ/bất đồng bộ với các mô hình ngôn ngữ lớn (LLM), quản lý các Prompt Templates và định dạng đầu ra (Output Parsers).

---

## 3. Các Mô Hình Ngôn Ngữ Lớn (LLM) & Embeddings

- **OpenAI API**: Sử dụng để gọi mô hình **`text-embedding-3-small`** với kích thước biểu diễn vector là **1536 chiều**, phục vụ việc biểu diễn ngữ nghĩa của tài liệu hướng dẫn (Guideline) và câu truy vấn người dùng.
- **Mistral AI API & SDK**:
  - Giao tiếp thông qua gói `langchain-mistralai` và `mistralai`.
  - Mô hình LLM của Mistral được cấu hình để thực hiện các nhiệm vụ:
    - **Router nội bộ** (`internal_router`): Nhận dạng câu hỏi có chứa chỉ số xét nghiệm/công thức cần xử lý hay không.
    - **Generator** (`final_answer`): Tổng hợp câu trả lời cuối cùng dựa trên các bằng chứng y khoa đã được truy xuất và kiểm tra an toàn.

---

## 4. Cơ Sở Dữ Liệu & Công Cụ Truy Xuất (Database & Search Engine)

- **Neon Postgres**: Cơ sở dữ liệu đám mây (Serverless Postgres) dùng làm kho lưu trữ chính.
- **pgvector**: Tiện ích mở rộng (extension) trên Postgres giúp lưu trữ trực tiếp các vector nhúng (embeddings) và thực hiện tìm kiếm tương đồng vector (Vector Similarity Search) qua phép tính khoảng cách Cosine (`<=>`).
- **asyncpg (0.31.0)**: Thư viện kết nối CSDL PostgreSQL bất đồng bộ tốc độ cao cho Python, giúp xử lý đồng thời nhiều truy vấn (Semantic Search, Keyword Search, Lexical Search) mà không gây nghẽn (blocking).
- **rank-bm25 (0.2.2)**: Thuật toán so khớp từ khóa thưa thớt (sparse retrieval) được sử dụng để xây dựng chỉ mục tìm kiếm BM25 nội bộ của tài liệu, hỗ trợ tăng cường độ chính xác cho việc so khớp từ khóa.

---

## 5. Xử Lý Tài Liệu & Nhận Dạng OCR

- **Mistral OCR**: Dịch vụ nhận diện ký tự quang học nâng cao từ Mistral AI giúp phân tích cấu trúc phức tạp của các file PDF y khoa chứa nhiều bảng biểu phức tạp và mã hóa chúng thành định dạng văn bản có cấu trúc sạch sẽ.
- **PyMuPDF (1.24.13)** & **PyPDF2 (3.0.1)**: Thư viện phân tích cú pháp, trích xuất text, số trang và quản lý dữ liệu thô từ file PDF tài liệu y khoa.
- **ReportLab (4.2.5)** & **Pillow (11.1.0)**: Công cụ tạo và xử lý tài liệu, hình ảnh cho các báo cáo y tế.

---

## 6. Xử Lý Giọng Nói & Âm Thanh (STT, TTS, VAD)

VitalAI tích hợp một hệ thống giao tiếp bằng giọng nói (Voice Mode) tự động hóa thông qua các công nghệ xử lý tín hiệu số cục bộ:

- **Voice Activity Detection (VAD)**:
  - **Silero VAD** (tải trực tiếp qua PyTorch Hub `snakers4/silero-vad`): Mô hình máy học nhỏ gọn, hiệu quả cao được dùng để phát hiện giọng nói của người dùng theo thời gian thực (Real-time Speech Detection) giúp hệ thống tự động nhận biết lúc bắt đầu nói và lúc kết thúc nói để ngắt luồng thu âm.
- **Audio Enhancement (Tăng cường âm thanh)**:
  - **DeepFilterNet** (`df.enhance`): Thư viện lọc nhiễu âm thanh sử dụng mạng nơ-ron sâu để làm sạch tiếng ồn môi trường trước khi chuyển tín hiệu âm thanh vào mô hình nhận dạng giọng nói.
- **Speech-to-Text (STT)**:
  - **PhoWhisper (`vinai/PhoWhisper-base`)**: Mô hình nhận dạng giọng nói tiếng Việt chuyên sâu của VinAI, chạy cục bộ bằng thư viện `transformers` của Hugging Face và kết hợp với `torchaudio` để chuyển hóa giọng nói của người dùng thành văn bản thô.
- **Text-to-Speech (TTS)**:
  - Sử dụng **Web Speech API** tích hợp sẵn trên các trình duyệt hiện đại để đọc to câu trả lời của AI.
  - Phía backend của VitalAI thiết lập một module tiền xử lý văn bản chuyên biệt (`prepare_tts_text`) để làm sạch toàn bộ định dạng Markdown, ký hiệu toán học, đồng thời chuyển hóa các chữ viết tắt y khoa (như *eGFR, ACR, CKD, KDIGO*) thành dạng phát âm tiếng Việt tự nhiên (ví dụ: *"mức lọc cầu thận ước tính"*, *"a c rờ"*, *"bệnh thận mạn"*, *"ca đi gô"*).

---

## 7. Khung Tính Toán Máy Học & Reranking Cục Bộ

- **PyTorch (>= 2.6.0)**: Nền tảng tính toán tensor và chạy các mô hình học sâu cục bộ (PhoWhisper, Silero VAD, DeepFilterNet). Hỗ trợ tăng tốc phần cứng thông qua GPU NVIDIA CUDA.
- **Sentence-Transformers (3.0.1)**: Sử dụng để chạy mô hình **Cross-Encoder Reranker** cục bộ. Mô hình này nhận vào câu hỏi và các chunks kết quả sau khi fusion, tính toán điểm số liên quan trực tiếp để chọn ra các mảnh thông tin thực sự giá trị cho câu trả lời.

---
*Tài liệu kỹ thuật - Đội ngũ phát triển VitalAI.*
