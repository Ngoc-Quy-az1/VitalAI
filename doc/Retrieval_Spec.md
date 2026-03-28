# Retrieval Spec

## Mục tiêu của tài liệu này

File này mô tả cách VitalAI sẽ thực hiện retrieval sau khi đã có output của phase ingestion.

Nó trả lời các câu hỏi:

1. query sẽ được hiểu như thế nào
2. metadata pre-filter hoạt động ra sao
3. hybrid retrieval sẽ được ghép như thế nào
4. evidence nào sẽ được đưa cho agent
5. khi nào phải retry hoặc fallback

Đây là file cầu nối giữa:

- `Ingestion_Spec.md`
- `Architecture.md`
- phase bắt đầu code retriever

## Input của retrieval layer

Retrieval layer không làm việc trực tiếp trên PDF.

Input của nó phải là dữ liệu đã được ingestion thành:

- `chunks.jsonl`
- `thresholds.jsonl`
- `formulas.json`

Về mặt logic, retrieval layer sẽ đọc từ:

- `medical_documents`
- `medical_thresholds`
- `medical_formulas`

## 2 loại truy vấn chính

### 1. Question query

Ví dụ:

- `GFR bao nhiêu thì được xem là bệnh thận mạn?`
- `Albumin niệu A3 là gì?`
- `Viêm cầu thận lupus điều trị như thế nào?`

Đặc điểm:

- có thể cần semantic context
- có thể cần threshold exact lookup
- có thể cần both

### 2. Lab interpretation query

Ví dụ:

- `ACR 350 mg/g, creatinin 2.1 mg/dL`

Đặc điểm:

- parse biomarker là bước bắt buộc
- exact lookup quan trọng hơn semantic search
- semantic retrieval chỉ bổ sung giải thích và citation

## Retrieval triết lý chung

VitalAI không nên làm retrieval theo thứ tự:

1. embed query
2. vector search
3. trả kết quả

Thứ tự đúng hơn là:

1. hiểu query
2. suy ra metadata intent
3. structured lookup nếu phù hợp
4. hybrid retrieval trên semantic corpus
5. hợp nhất evidence

## Bước 1 — Query understanding

Mục tiêu:

- xác định query type
- xác định disease hint
- xác định biomarker hint
- xác định section hint

### Output đề xuất

```json
{
  "query_type": "question",
  "disease_hint": "benh_than_man",
  "biomarker_hint": ["GFR"],
  "section_hint": "definition",
  "needs_structured_lookup": true,
  "needs_semantic_retrieval": true
}
```

## Bước 2 — Metadata pre-filter

Đây là bước quan trọng nhất để giảm retrieval noise.

### Các field nên dùng để pre-filter

- `disease_name`
- `section_type`
- `content_type`
- `biomarker`
- `doc_type`

### Ví dụ

#### Query

`GFR bao nhiêu thì là bệnh thận mạn?`

#### Pre-filter mong muốn

- `disease_name = benh_than_man`
- `biomarker = GFR`
- `section_type in [definition, classification]`

#### Query

`Albumin niệu A3 là gì?`

#### Pre-filter mong muốn

- `disease_name = benh_than_man`
- `biomarker = ACR`
- `section_type = classification`

#### Query

`Điều trị lupus nephritis`

#### Pre-filter mong muốn

- `disease_name = lupus_nephritis`
- `section_type = treatment`

## Bước 3 — Structured lookup

Structured lookup chạy trước semantic retrieval trong các trường hợp:

- query hỏi threshold
- query hỏi stage/rule
- input là lab result
- query hỏi công thức

### Structured lookup có thể trả về

1. `threshold_matches`
2. `formula_matches`
3. `classification_matches`

### Ví dụ threshold lookup

Query:

`GFR bao nhiêu thì là bệnh thận mạn?`

Expected match:

- `GFR < 60`
- `duration_condition = kéo dài trên 3 tháng`

### Ví dụ formula lookup

Query:

`GFR được tính theo công thức nào?`

Expected match:

- MDRD
- Cockcroft-Gault

## Bước 4 — Semantic retrieval

Sau metadata pre-filter, semantic retrieval mới có ý nghĩa.

### Semantic corpus

Nguồn semantic search chỉ nên lấy từ:

- `chunks.jsonl`
- tương đương `medical_documents`

### Mục tiêu của semantic retrieval

- lấy definition prose
- lấy disease context
- lấy treatment explanation
- lấy đoạn văn để citation và explanation

## Bước 5 — Hybrid retrieval

Hybrid retrieval gồm 2 thành phần:

### 1. Vector retrieval

Dùng cho:

- ngữ nghĩa
- paraphrase
- clinical explanation

### 2. FTS/BM25 retrieval

Dùng cho:

- biomarker
- tên thuốc
- stage label
- acronym
- exact phrase

### Tại sao cần hybrid

Chỉ dùng vector sẽ yếu ở:

- `GFR`
- `ACR`
- `KDIGO`
- `A1/A2/A3`
- exact threshold phrase

## Bước 6 — Rank fusion

Sau khi có kết quả từ vector và keyword retrieval, cần hợp nhất chúng.

### Mục tiêu

- giữ chunk có ngữ nghĩa đúng
- không bỏ sót chunk có keyword rất quan trọng

### Kết quả mong muốn

Tạo một danh sách candidate chunks có score tổng hợp.

Không cần khóa chặt công thức fusion ở phase docs, nhưng logic nên là:

- vector score
- keyword score
- metadata match bonus

## Bước 7 — Evidence assembly

Sau retrieval, hệ thống không nên chỉ trả về một list chunk thô.

Nó nên tạo `evidence bundle` chuẩn cho agent.

### Schema logic đề xuất

```json
{
  "query": "GFR bao nhiêu thì là bệnh thận mạn?",
  "structured_evidence": {
    "thresholds": [],
    "formulas": []
  },
  "semantic_evidence": [
    {
      "chunk_id": "ckd_p133_001",
      "content": "...",
      "metadata": {},
      "score": 0.91
    }
  ],
  "sources": [
    {
      "source_file": "ĐẠI CƯƠNG VỀ BỆNH LÝ CẦU THẬN( No Table) .pdf",
      "page": 133
    }
  ]
}
```

## Bước 8 — Relevance grading

Relevance grading là lớp kiểm soát chất lượng trước khi đưa evidence vào agent.

### Mục tiêu

- loại chunk nhiễu
- kiểm tra evidence có đủ để trả lời chưa
- quyết định có cần retry hay không

### Các trường hợp nên retry

- có chunk nhưng lệch disease
- có chunk nhưng lệch section
- query hỏi threshold nhưng không có structured match
- score thấp do query chưa rõ biomarker

### Retry strategy gợi ý

#### Retry 1

- mở rộng synonym

#### Retry 2

- hẹp metadata filter hơn

#### Retry 3

- tách query thành nhiều sub-query

## Bước 9 — Fallback strategy

Không phải query nào cũng nên cố trả lời bằng mọi giá.

### Nếu structured evidence thiếu

Và query là threshold-sensitive:

- nên trả lời thận trọng
- nói rõ dữ liệu hiện có không đủ chắc

### Nếu semantic evidence yếu

- không nên synthesis quá mạnh
- phải giảm mức độ khẳng định

### Nếu cả hai đều yếu

- nên refuse hoặc answer with limitation

## Retrieval strategy theo từng loại query

## A. Definition query

Ví dụ:

- `GFR bao nhiêu thì là bệnh thận mạn?`

Chiến lược:

1. threshold lookup
2. metadata pre-filter
3. hybrid retrieval
4. evidence assembly

## B. Classification query

Ví dụ:

- `Albumin niệu A3 là gì?`

Chiến lược:

1. classification rule lookup
2. metadata pre-filter vào `classification`
3. hybrid retrieval để lấy explanation

## C. Treatment query

Ví dụ:

- `Lupus nephritis điều trị như thế nào?`

Chiến lược:

1. disease + treatment pre-filter
2. hybrid retrieval
3. ít phụ thuộc structured lookup hơn

## D. Lab-result query

Ví dụ:

- `ACR 350 mg/g, creatinin 2.1 mg/dL`

Chiến lược:

1. parse lab values
2. threshold lookup
3. formula lookup
4. semantic retrieval để lấy explanation/source

## Retrieval output quality checklist

Retrieval được xem là đủ tốt khi:

- top evidence đúng disease
- top evidence đúng section
- threshold query có structured match
- source file và page luôn giữ được
- semantic chunk không bị mất ngữ cảnh

## Điều chưa làm ở phase này

Chưa làm:

- response synthesis
- safety layer
- API
- evaluation dataset

## Kết luận

Sau file này, bước tiếp theo hợp lý là:

- viết `Agent_Graph_Spec.md`

Vì từ đây đã có:

- data contract
- ingestion contract
- retrieval contract

Nghĩa là đã đủ nền để chốt input/output cho từng node trong Agentic RAG graph.
