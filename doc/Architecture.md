# Architecture

## Mục tiêu của tài liệu này

File này mô tả **kiến trúc mục tiêu** của VitalAI.

Đây chưa phải mô tả code hiện có. Nó dùng để:

- chốt ranh giới các module
- tránh code sai thứ tự
- giúp việc triển khai Agentic RAG có đường đi rõ ràng

## 2 use case bắt buộc phải support

### A. Medical question answering

Ví dụ:

- `Chỉ số GFR bao nhiêu thì được xem là bệnh thận mạn?`
- `Albumin niệu A3 nghĩa là gì?`

Yêu cầu:

- lấy đúng nguồn
- trả lời ngắn gọn
- luôn có citation

### B. Lab-result interpretation

Ví dụ:

- `ACR 350 mg/g, creatinin 2.1 mg/dL`

Yêu cầu:

- nhận diện biomarker
- so sánh với ngưỡng
- áp công thức nếu đủ dữ liệu đầu vào
- trả báo cáo tham khảo an toàn

## Quan điểm kiến trúc

Project này không nên được xây theo kiểu:

- mọi thứ đều là text chunk
- mọi thứ đều đưa vào vector DB
- agent chỉ retrieval rồi generate

Vì dữ liệu y khoa hiện có chứa:

- numeric thresholds
- classification rules
- formulas
- prose explanation

Do đó kiến trúc phải tách ra thành 2 lớp tri thức:

### 1. Semantic knowledge layer

Dùng cho:

- guideline
- disease explanation
- treatment prose
- context diễn giải

### 2. Structured knowledge layer

Dùng cho:

- threshold
- formula
- staging rule
- reference range

## Kiến trúc logic mục tiêu

```text
User
  |
  v
API / Interface Layer
  |
  v
Input Router
  |
  +--> Question Flow
  |      -> query understanding
  |      -> metadata pre-filter
  |      -> hybrid retrieval
  |      -> evidence grading
  |      -> response synthesis
  |
  +--> Lab Flow
         -> biomarker extraction
         -> threshold lookup
         -> formula lookup / calculation
         -> hybrid retrieval for explanation
         -> evidence grading
         -> response synthesis
```

## Agentic RAG workflow mục tiêu

VitalAI nên dùng agentic workflow theo các node logic sau:

1. `input_classifier`
   Xác định input là question, lab_record hay mixed.

2. `record_parser`
   Chỉ chạy khi input có dữ liệu xét nghiệm.

3. `query_rewriter`
   Bổ sung disease hint, biomarker synonym, section hint.

4. `retriever`
   Dùng metadata pre-filter + hybrid search.

5. `relevance_grader`
   Kiểm tra evidence có đủ tốt hay không.

6. `formula_or_threshold_engine`
   Chạy lookup exact rule và tính toán nếu cần.

7. `response_synthesizer`
   Gộp structured evidence và semantic evidence thành câu trả lời cuối.

8. `safety_checker`
   Kiểm tra độ an toàn, citation, mức độ chắc chắn.

Node-level contract và routing detail được chốt riêng tại:

- [Agent_Graph_Spec.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/Agent_Graph_Spec.md)

## Nguyên tắc routing

### Nếu là question

- ưu tiên retrieval
- structured lookup chỉ hỗ trợ nếu query hỏi về threshold/formula

### Nếu là lab input

- ưu tiên parse + structured lookup trước
- retrieval chỉ đóng vai trò giải thích và trích nguồn

## Những phần chưa nên làm quá sớm

Chưa nên đụng vào:

- LangGraph implementation chi tiết
- API streaming
- memory/session phức tạp
- frontend

cho đến khi:

- metadata schema rõ ràng
- ingestion output format rõ ràng
- structured store design rõ ràng

## Dependency giữa các phase

Kiến trúc này chỉ hợp lý nếu triển khai đúng thứ tự:

1. data audit
2. metadata design
3. ingestion design
4. retrieval design
5. agent workflow
6. API

## Kết luận

Kiến trúc đúng cho VitalAI là:

- `hybrid knowledge system`
- `agentic orchestration`
- `semantic + structured retrieval`

Chứ không phải chỉ là một chatbot gọi vector search.
