# Agent Graph Spec

## Mục tiêu của tài liệu này

File này chốt thiết kế logic cho Agentic RAG workflow của VitalAI.

Nó trả lời:

1. state của agent gồm những gì
2. mỗi node nhận gì và trả gì
3. routing condition giữa các node ra sao
4. khi nào retry
5. khi nào fallback hoặc stop

Đây là bước cầu nối giữa:

- `Retrieval_Spec.md`
- giai đoạn bắt đầu viết code graph và node logic

## Vai trò của agent trong hệ thống

Agent không thay thế:

- ingestion
- retrieval
- structured lookup

Agent chỉ làm orchestration cho các thành phần đó.

Nói cách khác:

- ingestion tạo dữ liệu
- retrieval lấy evidence
- agent quyết định dùng evidence nào và synthesis theo flow nào

## 3 loại input chính

### 1. `question`

Ví dụ:

- `GFR bao nhiêu thì là bệnh thận mạn?`

### 2. `lab_record`

Ví dụ:

- `ACR 350 mg/g, creatinin 2.1 mg/dL`

### 3. `mixed`

Ví dụ:

- `ACR 350 mg/g có phải A3 không?`

## Nguyên tắc graph design

### 1. Không để graph quá thông minh ở sai chỗ

Graph không nên gánh phần:

- OCR
- PDF parsing
- retrieval indexing

Graph chỉ nên xử lý:

- classify
- route
- call retrieval
- call structured lookup
- synthesize
- safety check

### 2. Structured-first cho lab input

Nếu input là `lab_record` hoặc `mixed`, graph phải ưu tiên:

1. parse values
2. structured lookup
3. semantic retrieval

chứ không được làm ngược lại.

### 3. Retry có giới hạn

Graph phải có retry, nhưng không được loop vô hạn.

## State schema logic

### Trạng thái tối thiểu

```json
{
  "session_id": "string",
  "raw_input": "string",
  "input_type": "question | lab_record | mixed",
  "query": "string",
  "rewritten_query": "string | null",
  "disease_hint": "string | null",
  "biomarker_hints": [],
  "section_hint": "string | null",
  "parsed_lab_values": [],
  "structured_evidence": {
    "thresholds": [],
    "formulas": []
  },
  "semantic_evidence": [],
  "evidence_bundle": {},
  "retrieval_score": 0.0,
  "retry_count": 0,
  "final_answer": "string | null",
  "citations": [],
  "safety_flags": []
}
```

## Field quan trọng nhất trong state

### Input understanding

- `raw_input`
- `input_type`
- `query`

### Query understanding

- `rewritten_query`
- `disease_hint`
- `biomarker_hints`
- `section_hint`

### Structured processing

- `parsed_lab_values`
- `structured_evidence`

### Retrieval processing

- `semantic_evidence`
- `evidence_bundle`
- `retrieval_score`
- `retry_count`

### Output

- `final_answer`
- `citations`
- `safety_flags`

## Danh sách node đề xuất

### 1. `input_classifier`

#### Mục tiêu

Phân loại input thành:

- `question`
- `lab_record`
- `mixed`

#### Input

- `raw_input`

#### Output

- `input_type`
- `query`

## 2. `query_analyzer`

#### Mục tiêu

Rút ra hint cho retrieval:

- disease
- biomarker
- section

#### Input

- `query`
- `input_type`

#### Output

- `disease_hint`
- `biomarker_hints`
- `section_hint`

## 3. `record_parser`

#### Mục tiêu

Chỉ chạy cho:

- `lab_record`
- `mixed`

#### Input

- `raw_input`

#### Output

- `parsed_lab_values`

## 4. `query_rewriter`

#### Mục tiêu

Tạo query tốt hơn cho retrieval:

- expand synonym
- thêm biomarker alias
- thêm disease context nếu cần

#### Input

- `query`
- `disease_hint`
- `biomarker_hints`
- `section_hint`

#### Output

- `rewritten_query`

## 5. `structured_lookup`

#### Mục tiêu

Lấy evidence exact từ:

- threshold store
- formula store

#### Input

- `input_type`
- `query`
- `parsed_lab_values`
- `disease_hint`
- `biomarker_hints`

#### Output

- `structured_evidence.thresholds`
- `structured_evidence.formulas`

## 6. `semantic_retriever`

#### Mục tiêu

Gọi retrieval layer để lấy semantic evidence.

#### Input

- `rewritten_query`
- `disease_hint`
- `biomarker_hints`
- `section_hint`

#### Output

- `semantic_evidence`

## 7. `evidence_assembler`

#### Mục tiêu

Gộp:

- structured evidence
- semantic evidence

thành một `evidence_bundle` chuẩn cho bước grading và synthesis.

#### Output

- `evidence_bundle`
- `citations`

## 8. `relevance_grader`

#### Mục tiêu

Đánh giá retrieval hiện tại có đủ tốt không.

#### Input

- `evidence_bundle`

#### Output

- `retrieval_score`
- `safety_flags` nếu evidence yếu

## 9. `response_synthesizer`

#### Mục tiêu

Sinh câu trả lời cuối cùng dựa trên evidence đã được chấp nhận.

#### Input

- `input_type`
- `parsed_lab_values`
- `evidence_bundle`

#### Output

- `final_answer`

## 10. `safety_checker`

#### Mục tiêu

Kiểm tra:

- có citation chưa
- có nói quá độ phủ dữ liệu không
- có kết luận quá mạnh không
- có cảnh báo cần thiết chưa

#### Output

- `safety_flags`
- `final_answer` đã được chỉnh lại nếu cần

## Routing logic tổng thể

```text
START
  |
  v
input_classifier
  |
  v
query_analyzer
  |
  +--> if input_type in [lab_record, mixed]
  |        -> record_parser
  |        -> query_rewriter
  |        -> structured_lookup
  |        -> semantic_retriever
  |
  +--> if input_type == question
           -> query_rewriter
           -> structured_lookup (optional)
           -> semantic_retriever
  |
  v
evidence_assembler
  |
  v
relevance_grader
  |
  +--> if score low and retry_count < max_retry
  |        -> query_rewriter
  |        -> semantic_retriever
  |
  +--> else
           -> response_synthesizer
           -> safety_checker
           -> END
```

## Retry logic

### Khi nào retry

Nên retry nếu:

- semantic evidence đúng domain nhưng quá yếu
- query thiếu synonym
- query thiếu disease context
- query hỏi threshold nhưng retrieval chưa lấy được explanation chunk phù hợp

### Không nên retry nếu

- structured evidence đã đủ mạnh
- dữ liệu nền không có coverage cho câu hỏi
- query nằm ngoài phạm vi nephrology của corpus

### `max_retry`

Đề xuất:

- `max_retry = 2` hoặc `3`

Không nên cao hơn ở giai đoạn đầu.

## Fallback logic

### Case 1 — Structured đủ, semantic yếu

Hệ thống vẫn có thể trả lời, nhưng:

- câu trả lời ngắn
- mức độ khẳng định thấp hơn
- nói rõ dữ liệu giải thích còn hạn chế

### Case 2 — Semantic đủ, structured không có

Nếu query không phải threshold-sensitive:

- có thể vẫn trả lời

Nếu query là threshold-sensitive:

- không nên trả lời mạnh

### Case 3 — Cả hai đều yếu

- nên refuse hoặc answer with limitation

## Safety rules ở graph level

Graph phải đảm bảo:

1. câu trả lời luôn có citation nếu có evidence
2. threshold-sensitive query không được trả lời chỉ bằng semantic guess
3. lab interpretation không được bỏ qua structured lookup
4. nếu dữ liệu không đủ, phải nói là không đủ

## Output cuối cùng của graph

Graph nên trả ra object logic như sau:

```json
{
  "input_type": "mixed",
  "final_answer": "...",
  "citations": [
    {
      "source_file": "ĐẠI CƯƠNG VỀ BỆNH LÝ CẦU THẬN( No Table) .pdf",
      "page": 133
    }
  ],
  "structured_evidence": {
    "thresholds": [],
    "formulas": []
  },
  "safety_flags": []
}
```

## Điều chưa làm ở phase này

Chưa làm:

- prompt wording chi tiết
- implementation cụ thể bằng LangGraph
- API request/response schema
- persistence của session

## Kết luận

Sau file này, bước hợp lý tiếp theo là:

- viết `API_Spec.md`

hoặc

- chốt prompt/spec cho từng node nếu muốn đi sâu thêm trước khi code.
