# Agentic RAG Implementation Plan

## Mục tiêu của tài liệu này

Đây là file plan trung tâm cho toàn bộ project.

Nó trả lời:

- sẽ làm gì trước
- sẽ không làm gì quá sớm
- mỗi phase cần bàn giao thứ gì

## Nguyên tắc điều phối

Project này nên đi theo thứ tự:

1. `understand data`
2. `design contract`
3. `build ingestion`
4. `build retrieval`
5. `build agent`
6. `build API`

Không nên đảo thứ tự.

## Phase 0 — Clarify scope and data

### Mục tiêu

Chốt phạm vi thực tế của project dựa trên dữ liệu hiện có.

### Việc cần làm

- kiểm kê tất cả file nguồn
- xác định tài liệu nào là raw, tài liệu nào là processed
- xác định domain coverage hiện có
- xác định use case nào có thể support ngay, use case nào chưa

### Deliverable

- inventory tài liệu
- taxonomy tài liệu
- cập nhật lại `Data_Audit.md`

### Exit criteria

- biết chính xác đang có bao nhiêu nguồn
- biết rõ hiện tại project chỉ mạnh ở nephrology hay đã đủ cho lab chatbot rộng hơn

## Phase 1 — Metadata and storage contract

### Mục tiêu

Chốt data contract trước khi viết ingestion.

### Việc cần làm

- chốt `doc_type`
- chốt `disease_name`
- chốt `section_type`
- chốt `content_type`
- chốt fields cho threshold và formula
- chốt logical DB design

### Deliverable

- metadata schema draft
- structured record schema cho threshold
- structured record schema cho formula
- file contract riêng cho output: `chunks`, `thresholds`, `formulas`
- cập nhật `Database.md`

### Exit criteria

- mọi người thống nhất được mỗi loại dữ liệu sẽ đi vào bảng nào
- mọi người thống nhất được field nào là bắt buộc, field nào là optional

## Phase 2 — Ingestion design

### Mục tiêu

Thiết kế pipeline chuyển PDF hỗn hợp thành dữ liệu có thể index được.

### Việc cần làm

- thiết kế content classifier
- thiết kế prose chunking strategy
- thiết kế threshold extractor
- thiết kế JSON-block salvage strategy
- thiết kế output format

### Deliverable

- ingestion spec
- chunk JSONL spec
- threshold JSONL spec
- formula JSON spec
- validation rule cho từng output

### Exit criteria

- có thể mô tả rõ một file PDF sẽ được chuyển thành những output nào
- có thể mô tả rõ block nào bị reject và block nào phải review thủ công

## Phase 3 — Retrieval design

### Mục tiêu

Chốt retrieval pipeline cho Agentic RAG.

### Việc cần làm

- chốt metadata pre-filter logic
- chốt hybrid search logic
- chốt citation strategy
- chốt relevance grading strategy

### Deliverable

- retrieval flow
- ranking strategy
- definition của evidence bundle dùng cho agent
- retry/fallback strategy

### Exit criteria

- có thể giải thích rõ với mỗi query, hệ thống lấy evidence theo thứ tự nào
- có thể giải thích rõ khi nào retrieval được xem là không đủ tốt để trả lời mạnh

## Phase 4 — Agent workflow design

### Mục tiêu

Chốt graph logic trước khi code agent.

### Node đề xuất

1. `input_classifier`
2. `record_parser`
3. `query_rewriter`
4. `retriever`
5. `relevance_grader`
6. `formula_or_threshold_engine`
7. `response_synthesizer`
8. `safety_checker`

### Việc cần làm

- xác định input/output của từng node
- xác định routing condition
- xác định retry condition
- xác định safety condition

### Deliverable

- graph spec
- state spec
- node contract spec
- retry/fallback logic ở graph level

### Exit criteria

- có thể implement graph mà không cần tranh luận lại logic tổng thể
- có thể giải thích rõ mỗi loại input sẽ đi qua node nào

## Phase 5 — Implementation order

Khi bắt đầu code, nên code theo thứ tự:

1. config và constants
2. metadata schema
3. ingestion parser
4. structured extractors
5. chunk output writer
6. retrieval layer
7. agent nodes
8. API
9. evaluation

### API contract cần chốt trước khi code backend

- endpoint tối thiểu
- request schema
- response schema
- citation field
- safety field
- error handling

## Phase 6 — Evaluation and safety

### Mục tiêu

Đảm bảo hệ thống không chỉ chạy được mà còn an toàn.

### Việc cần làm

- tạo gold dataset
- test retrieval recall
- test threshold accuracy
- test formula accuracy
- test citation completeness
- test refusal/safety behavior

### Exit criteria

- trả lời có nguồn
- threshold không bị hallucinate
- câu trả lời không vượt quá độ phủ dữ liệu

## Điều chưa nên làm ngay

Chưa nên làm ở thời điểm hiện tại:

- frontend
- voice
- memory phức tạp
- session DB chi tiết
- streaming

Lý do:

- đây không phải critical path
- chưa giải quyết xong data foundation

## Ưu tiên thực tế cho turn tiếp theo

Nếu làm tiếp từng bước để dễ hiểu, thứ tự hợp lý là:

1. chốt `inventory dữ liệu`
2. chốt `taxonomy tài liệu`
3. chốt `metadata schema`
4. chốt `ingestion spec`
5. chốt `retrieval spec`
6. chốt `agent graph spec`

Sau sáu việc này mới nên bắt đầu viết code ingestion, retrieval, và graph.
