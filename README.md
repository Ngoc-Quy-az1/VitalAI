# VitalAI

VitalAI hiện đang ở giai đoạn **planning và documentation**, chưa bắt đầu triển khai backend hay Agentic RAG.

Mục tiêu dài hạn của project:

1. Trả lời câu hỏi y khoa có trích dẫn nguồn rõ ràng.
2. Đọc kết quả xét nghiệm, đối chiếu ngưỡng tham chiếu, áp công thức khi cần, rồi sinh báo cáo tham khảo an toàn.

## Trạng thái hiện tại

### Đã có

- bộ tài liệu định hướng
- phân tích sơ bộ bộ dữ liệu đang có trong repo
- thiết kế kiến trúc mục tiêu
- kế hoạch triển khai Agentic RAG theo phase

### Chưa có

- ingestion pipeline
- retrieval pipeline
- structured threshold/formula store
- agent graph
- API backend
- evaluation dataset

## Điều quan trọng nhất cần hiểu

Repo này hiện **chưa phải codebase production**.

Đây là nơi để:

- hiểu dữ liệu thật sự đang có gì
- chốt thiết kế hệ thống trước khi code
- thống nhất thứ tự triển khai

Không nên hiểu nhầm rằng:

- đã có agent chạy được
- đã có schema DB chính thức
- đã có vector DB hay API sẵn sàng dùng

## Bộ tài liệu nên đọc theo thứ tự

1. [Data_Audit.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/Data_Audit.md)
2. [RAG_Design.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/RAG_Design.md)
3. [Metadata_Schema.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/Metadata_Schema.md)
4. [Ingestion_Spec.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/Ingestion_Spec.md)
5. [Retrieval_Spec.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/Retrieval_Spec.md)
6. [Agent_Graph_Spec.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/Agent_Graph_Spec.md)
7. [API_Spec.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/API_Spec.md)
8. [Architecture.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/Architecture.md)
9. [Database.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/Database.md)
10. [Implementation_Plan.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/Implementation_Plan.md)

## Tóm tắt dữ liệu hiện có

Hiện mới xác nhận một file chính:

- `data/raw_data/ĐẠI CƯƠNG VỀ BỆNH LÝ CẦU THẬN( No Table) .pdf`

Vấn đề:

- file hiện đang nằm ở `raw_data`, đây là naming hợp lý hơn
- `processed_data` hiện vẫn trống và chưa chứa output thực sự
- nội dung trong PDF trộn nhiều loại:
  - prose y khoa
  - ngưỡng số liệu inline
  - JSON nhúng
  - công thức và bảng phân loại

Hệ quả:

- không thể bắt đầu bằng fixed-size chunking
- không thể dùng vector-only RAG
- phải thiết kế dual-storage: semantic + structured

## Định hướng Agentic RAG

Hệ thống mục tiêu sẽ có 2 luồng chính:

### 1. Question answering

Luồng này dùng cho các câu hỏi như:

- `GFR bao nhiêu thì được xem là bệnh thận mạn?`
- `Albumin niệu A3 là gì?`

### 2. Lab interpretation

Luồng này dùng cho input như:

- `ACR 350 mg/g, creatinin 2.1 mg/dL`

Luồng này bắt buộc phải:

- parse biomarker
- tra threshold structured
- tra công thức structured nếu cần
- dùng retrieval để lấy context giải thích

## Roadmap ở mức cao

### Phase 0 — Clarify data contract

- xác nhận tất cả tài liệu nguồn thực có
- đổi lại naming raw/processed cho đúng
- chốt taxonomy tài liệu

### Phase 1 — Design metadata and storage

- chốt metadata schema
- chốt structured fields cho threshold/formula
- chốt logical DB design

### Phase 2 — Build ingestion

- content classifier
- prose chunker
- threshold extractor
- JSON block parser
- output validation

### Phase 3 — Build retrieval

- metadata pre-filter
- hybrid search
- citation-ready evidence assembly
- retry/fallback logic

### Phase 4 — Build agent workflow

- input classifier
- record parser
- retriever
- formula engine
- synthesizer
- safety checker

### Phase 5 — API and evaluation

- API
- gold dataset
- retrieval evaluation
- response safety checks
- error/limitation contract

## Nguyên tắc làm việc cho repo này

- đi từng bước nhỏ
- chốt docs trước khi code
- không code retrieval khi data contract còn mơ hồ
- không hứa use case vượt quá độ phủ của dữ liệu

## Kết luận

Việc đúng cần làm tiếp không phải là viết code ngay.

Việc đúng là dùng bộ docs hiện tại để thống nhất:

1. dữ liệu nào đang có thật
2. output data sẽ có schema gì
3. ingestion sẽ route từng loại block như thế nào
4. retrieval sẽ hoạt động theo contract nào
5. hệ thống sẽ được tách thành những thành phần nào
6. phase nào phải xong trước khi bước sang phase tiếp theo
