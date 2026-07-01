# Nền Tảng Lý Thuyết và Kiến Trúc Kỹ Thuật Dự Án VitalAI

Tài liệu này cung cấp mô tả chi tiết, toàn diện về các nền tảng lý thuyết, mô hình kiến trúc, phương pháp xử lý dữ liệu và thuật toán được áp dụng trong dự án **VitalAI** nhằm giải quyết bài toán trợ lý ảo y khoa (tập trung vào chuyên khoa Thận học - Nephrology).

---

## 1. Kiến Trúc Tổng Quan: Hybrid Knowledge System (Hệ thống Tri thức Lai)

Trong lĩnh vực y tế, một hệ thống RAG (Retrieval-Augmented Generation) thông thường chỉ dựa vào cơ sở dữ liệu vector (Vector Database) sẽ dễ gặp phải các hạn chế nghiêm trọng:
- **Sai lệch số liệu (Numerical Hallucination)**: LLM tự tính toán hoặc suy diễn sai các ngưỡng chẩn đoán (ví dụ: mức lọc cầu thận eGFR, tỷ lệ albumin/creatinin niệu ACR).
- **Mất ngữ cảnh có cấu trúc**: Các công thức y khoa phức tạp hoặc bảng phân giai đoạn bệnh bị phân mảnh thành các đoạn text rời rạc (prose chunks) dẫn đến việc truy xuất không chính xác.

Để giải quyết triệt để vấn đề này, VitalAI áp dụng kiến trúc **Hệ thống Tri thức Lai (Hybrid Knowledge System)** tách biệt thông tin thành hai lớp rõ rệt:

```text
                      +---------------------------------------+
                      |             User Query                |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         Input Classifier & Router     |
                      +---------------------------------------+
                                     /         \
                                    /           \
                                   v             v
       +----------------------------------+       +----------------------------------+
       |   Semantic Knowledge Layer       |       |   Structured Knowledge Layer     |
       |   - Tài liệu Guideline dạng văn bản |       |   - Chỉ số Xét nghiệm (Threshold)|
       |   - Sách chuyên khoa, phác đồ điều trị|   |   - Công thức Y khoa (Formula)   |
       +----------------------------------+       +----------------------------------+
                                   \             /
                                    \           /
                                     v         v
                      +---------------------------------------+
                      |        Response Synthesizer           |
                      +---------------------------------------+
```

1. **Lớp Tri thức Ngữ nghĩa (Semantic Knowledge Layer)**: Chứa các kiến thức mô tả, định nghĩa, hướng dẫn lâm sàng dưới dạng phi cấu trúc. Được quản lý bằng Vector DB và công cụ tìm kiếm toàn văn (Full-Text Search).
2. **Lớp Tri thức Cấu trúc (Structured Knowledge Layer)**: Chứa các quy tắc, ngưỡng số liệu tĩnh (Thresholds) và các công thức y học (Formulas) được số hóa dưới dạng mã nguồn logic lập trình và cơ sở dữ liệu quan hệ (Deterministic Evaluation).

---

## 2. Quy Trình Xử Lý và Phân Loại Dữ Liệu Đầu Vào (Data Ingestion Pipeline)

Để đảm bảo dữ liệu đưa vào công cụ tìm kiếm không bị nhiễu chéo và giữ nguyên vẹn cấu trúc y tế, VitalAI không áp dụng kỹ thuật phân mảnh kích thước cố định (fixed-size chunking) thông thường. Thay vào đó, quy trình Ingestion thực hiện các bước sau:

### Phân loại Nội dung Trước khi Phân mảnh (Content Classification Before Chunking)
Mỗi khối dữ liệu trong tài liệu nguồn (PDF/JSON) trước khi đưa vào cơ sở dữ liệu sẽ được phân loại thành một trong các nhóm:
- `prose`: Các đoạn văn bản giải nghĩa thông thường.
- `threshold_value`: Các câu hoặc bảng chứa ngưỡng chỉ số sinh học (ví dụ: `protein niệu > 3.5 g/24h`).
- `json_block`: Cấu trúc dữ liệu bán cấu trúc chứa bảng biểu hoặc phân loại.
- `formula`: Các biểu thức toán học y khoa.
- `table_like`: Các thông tin dạng hàng và cột.

### Biểu diễn Vector Ngữ nghĩa (Dense Vector Representation)
- **Embedding Model**: Sử dụng mô hình `text-embedding-3-small` của OpenAI để chuyển đổi các đoạn văn bản (chunks) thành các vector biểu diễn ngữ nghĩa với kích thước **1536 chiều**.
- **Vector Database**: Sử dụng extension **pgvector** chạy trên nền tảng cơ sở dữ liệu **Neon/Postgres**. Phép toán so sánh tương đồng ngữ nghĩa sử dụng **Cosine Distance** (`<=>` trong pgvector):
  $$\text{Cosine Similarity} = 1 - (\text{embedding} \Leftrightarrow \text{query\_embedding})$$

---

## 3. Kiến Trúc Truy Xuất Lai (Hybrid Retrieval) & Thuật Toán Fusion

Để tối ưu hóa khả năng tìm kiếm cả về mặt ngữ nghĩa (semantic) lẫn các thuật ngữ chuyên ngành tiếng Việt, từ viết tắt và chỉ số xét nghiệm (lexical/keyword), VitalAI sử dụng bộ truy xuất **3 nhánh kết hợp** song song:

```text
                  +---------------------------------------------+
                  |               Retrieval Plan                |
                  +---------------------------------------------+
                     /                   |                   \
                    /                    |                    \
                   v                     v                     v
      +-------------------+    +-------------------+    +-------------------+
      |  Semantic Search  |    | Full-Text Search  |    |  Lexical Search   |
      |   (pgvector DB)   |    | (Postgres FTS)    |    |   (ILIKE Match)   |
      +-------------------+    +-------------------+    +-------------------+
                    \                    |                    /
                     \                   |                   /
                      v                  v                  v
                  +---------------------------------------------+
                  |         Reciprocal Rank Fusion (RRF)        |
                  +---------------------------------------------+
                                         |
                                         v
                  +---------------------------------------------+
                  |      Parent/Neighbor Context Expansion      |
                  +---------------------------------------------+
                                         |
                                         v
                  +---------------------------------------------+
                  |          Reranker (Cross-Encoder)           |
                  +---------------------------------------------+
```

### Nhánh 1: Semantic Search (Tìm kiếm Ngữ nghĩa)
- So khớp vector truy vấn của người dùng với vector đại diện của các chunks trong bảng `medical_documents` bằng pgvector.
- Phù hợp để tìm các khái niệm diễn giải tương đồng nhưng khác biệt về mặt câu chữ.

### Nhánh 2: Keyword Search (PostgreSQL Full-Text Search)
- Sử dụng cấu hình `simple` của Postgres FTS để thực hiện tách từ và so khớp từ khóa:
  ```sql
  websearch_to_tsquery('simple', query) @@ to_tsvector('simple', content)
  ```
- Việc dùng parser `simple` giúp giữ nguyên vẹn các cụm từ chuyên ngành y học tiếng Việt mà không bị thuật toán cắt từ (stemming) của tiếng Anh làm sai lệch cấu trúc từ gốc.

### Nhánh 3: Lexical Search (Vietnamese-aware Substring Match)
- Sử dụng phép so khớp chuỗi con không phân biệt hoa thường (`ILIKE`) trên các trường `content`, `source_id`, và `metadata` của Postgres.
- **Lý do**: Postgres FTS `simple` đôi khi hoạt động không tốt với tiếng Việt do thiếu tokenizer tối ưu riêng biệt và không chuẩn hóa dấu tự động. Nhánh Lexical này giúp bắt chính xác các cụm từ quan trọng như: *"hội chứng thận hư"*, *"bệnh cầu thận thay đổi tối thiểu"*, *"KDIGO 2012"*, hoặc các chữ viết tắt như *"ACR"*, *"GFR"*.

### Nhánh Phụ: Local BM25 Search
- Hệ thống hỗ trợ tích hợp thêm bộ truy xuất BM25 (Best Matching 25) chạy trực tiếp trên index cục bộ của server nhằm bổ sung thêm trọng số từ khóa tần suất xuất hiện tài liệu.

### Thuật toán Hợp nhất Kết quả: Reciprocal Rank Fusion (RRF)
RRF được sử dụng để xếp hạng lại các kết quả trả về từ nhiều chiến lược truy xuất khác nhau mà không cần chuẩn hóa điểm số gốc của từng hệ thống về cùng một thang đo. Công thức tính điểm RRF cho một tài liệu $d$:
$$RRF(d) = \sum_{m \in M} \frac{W_m}{K + r_m(d)}$$

Trong đó:
- $M$ là tập hợp các nhánh truy xuất song song (Semantic, Keyword, Lexical, BM25).
- $r_m(d)$ là thứ hạng (rank) của tài liệu $d$ trong danh sách kết quả của nhánh $m$ (bắt đầu từ 1). Nếu tài liệu không xuất hiện, hạng sẽ coi là vô cùng lớn.
- $K$ là hằng số làm mượt (smoothing constant), trong dự án được thiết lập $K = 60$.
- $W_m$ là trọng số (weight) của từng nhánh cụ thể để tối ưu hóa độ chính xác:
  - `VECTOR_RRF_WEIGHT = 0.85` (Semantic)
  - `KEYWORD_RRF_WEIGHT = 1.0` (Postgres FTS)
  - `LEXICAL_RRF_WEIGHT = 1.15` (Vietnamese Substring)
  - `BM25_RRF_WEIGHT = 0.95` (Local BM25)

Sau khi tính điểm RRF cơ bản, tài liệu còn được cộng thêm các điểm thưởng phạt logic (`metadata_bonus` và `lexical_bonus`) trước khi sắp xếp thứ hạng cuối cùng.

### Kỹ thuật Mở rộng Ngữ cảnh Lân cận (Parent/Neighbor Context Expansion)
- **Vấn đề**: Tài liệu PDF y học khi cắt nhỏ thường chứa câu trả lời nằm ở mảnh (chunk) $N$ nhưng tiêu đề lớn đại diện cho ngữ cảnh đó lại nằm ở mảnh $N-1$ hoặc trang trước.
- **Giải pháp**: Khi một chunk được chọn từ truy xuất, VitalAI tự động truy vấn thêm các chunk lân cận (lùi lại 1 chunk, tiến lên tối đa 4 chunk) trên cùng một trang tài liệu dựa trên chỉ mục chunk (`chunk_index`) và mã file nguồn (`source_file`). Việc này giúp LLM có đầy đủ thông tin bối cảnh để đưa ra câu trả lời chính xác, tránh hiện tượng sinh thông tin sai lệch do mất tiêu đề lớn.

### Reranking (Xếp hạng lại)
- Dự án hỗ trợ tích hợp **Local Reranker (Cross-Encoder)** để đánh giá mức độ liên quan trực tiếp giữa cặp câu hỏi - câu trả lời ứng viên, hỗ trợ đắc lực cho việc lọc bỏ các thông tin rác trước khi đưa vào Prompt của LLM.

---

## 4. Điều Phối Luồng Agent Bằng LangGraph

Hệ thống sử dụng **LangGraph** để xây dựng đồ thị trạng thái (State Graph) điều phối luồng thông tin xử lý câu hỏi y khoa dưới dạng một máy trạng thái hữu hạn tuần tự (Directed Acyclic Graph - DAG) có điều kiện.

```text
    [START] ---> prepare_input ---> extract_tool_payload
                                           |
                    +----------------------+----------------------+
                    |                                             | (direct query: chào hỏi,...)
                    | (medical / RAG query)                       v
                    v                                       build_prompt
        route_with_medical_tools                                  |
                    |                                             v
                    v                                     generate_response
           call_medical_tools                                     |
                    |                                             v
                    v                                      cleanup_response
        understand_retrieval_query                                |
                    |                                             v
                    v                                           [END]
            retrieve_context ---> build_prompt ---> generate_response
```

### Các Node và Logic chính trong Graph

1. **`prepare_input` & `extract_tool_payload`**:
   - Chuẩn hóa đầu vào của người dùng.
   - Sử dụng một bộ phân tích cú pháp tĩnh (deterministic parser) để trích xuất sớm các chỉ số sinh học (như GFR, ACR, Creatinine,...) hoặc các biến công thức (tuổi, giới tính, cân nặng,...) từ văn bản thô của người dùng.
2. **`route_with_medical_tools`**:
   - Sử dụng các luật heuristic tĩnh hoặc LLM Router để quyết định xem câu hỏi có chứa chỉ số thực tế cần chạy tính toán logic của y tế (Medical Tools Service) hay chỉ cần truy xuất thông tin văn bản thông thường (RAG-only).
3. **`call_medical_tools`**:
   - Thực hiện gọi dịch vụ công cụ y tế để so khớp các ngưỡng chẩn đoán lâm sàng thực tế và tính toán công thức (ví dụ: tính eGFR bằng phương pháp CKD-EPI 2021, tính độ thải Creatinine Cockcroft-Gault, tính phân số thải Natri FENa). Kết quả trả về dưới dạng JSON cấu trúc.
4. **`understand_retrieval_query` (Query Planner)**:
   - Tổng hợp thông tin từ câu hỏi gốc của người dùng, kết quả phân tích của công cụ y tế, và các filters cấu hình sẵn để xây dựng một **Kế hoạch truy xuất (Retrieval Plan)** an toàn.
   - Đưa ra các bộ lọc cứng (hard filters) hoặc gợi ý mềm (soft hints) để tăng độ chuẩn xác truy xuất.
5. **`retrieve_context`**:
   - Thực hiện quy trình Hybrid Retrieval ở mục 3 để thu thập bằng chứng (evidence) từ Vector DB và Postgres FTS.
6. **`build_prompt` & `generate_response`**:
   - Tích hợp bằng chứng văn bản (Semantic context) và kết quả chẩn đoán chính xác của công cụ tính toán (Structured context) vào Prompt.
   - LLM (Mistral/GPT) sinh câu trả lời tự nhiên dưới dạng ngôn ngữ y khoa chuyên nghiệp và dễ hiểu.
7. **`cleanup_response`**:
   - Hậu xử lý văn bản, dọn dẹp các thẻ tag hệ thống, chuẩn hóa định dạng trích dẫn (citation) an toàn hiển thị lên giao diện người dùng.

---

## 5. Hệ Thống Tính Toán Công Thức và Ngưỡng Y Khoa (Medical Tools Service)

Để đảm bảo tính chính xác tuyệt đối của các phép tính y tế, VitalAI không để LLM tự thực hiện các phép toán số học. Toàn bộ các công thức được tính toán bằng mã nguồn Python deterministic trong lớp `MedicalToolsService` hỗ trợ:
- **CKD-EPI 2021 Creatinine**: Công thức chuẩn hóa quốc tế ước tính mức lọc cầu thận (eGFR) không phụ thuộc vào yếu tố chủng tộc (chỉ dựa vào Creatinine huyết thanh, tuổi và giới tính).
- **MDRD GFR**: Công thức ước tính chức năng thận truyền thống.
- **Cockcroft-Gault**: Đo lường độ thanh thải Creatinine phục vụ việc chỉnh liều thuốc trên lâm sàng.
- **FENa (Fractional Excretion of Sodium)**: Phân số thải natri giúp chẩn đoán phân biệt nguyên nhân gây suy thận cấp (trước thận vs tại thận).
- **BSA (Body Surface Area)**: Tính diện tích bề mặt cơ thể theo công thức Du Bois để hiệu chỉnh chỉ số.

---

## 6. Khung Đánh Giá Hệ Thống RAG (RAG Evaluation Framework)

Để kiểm soát chất lượng câu trả lời và hạn chế tối đa sai sót y học trước khi đưa hệ thống vào thực tế, VitalAI áp dụng quy trình đánh giá tự động dựa trên nền tảng **LangSmith RAG Evaluation** với bộ dữ liệu kiểm thử y khoa chất lượng cao.

Các chỉ số (metrics) đánh giá cốt lõi bao gồm:
- **Answer Correctness (Độ chính xác của câu trả lời)**: Đánh giá câu trả lời của AI so với đáp án chuẩn (ground truth) của chuyên gia y tế về mặt ngữ nghĩa và số liệu.
- **Context Recall (Độ phủ của ngữ cảnh)**: Đo lường xem hệ thống truy xuất (retriever) có lấy ra đầy đủ các chunks chứa thông tin cần thiết để trả lời câu hỏi hay không.
- **Faithfulness / Groundedness (Độ trung thực - Chống ảo giác)**: Đánh giá xem câu trả lời được sinh ra bởi LLM có hoàn toàn dựa trên ngữ cảnh đã được truy xuất hay tự ý bịa đặt thông tin.

---
*Bản quyền tài liệu thuộc về nhóm nghiên cứu phát triển đồ án tốt nghiệp VitalAI.*
