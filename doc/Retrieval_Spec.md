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

### Implementation update 2026-05-25

Query understanding hiện được triển khai bằng node:

- `understand_retrieval_query` trong `src/LLM/qa/graph.py`
- helper chính: `build_retrieval_plan(...)` trong `src/LLM/retrieval/query_planner.py`

Output thực tế là `retrieval_plan`:

```json
{
  "strategy": "deterministic_tool_aware_query_planner_v1",
  "query_type": "medical_qa",
  "query": "câu hỏi đã enrich",
  "filters": {
    "disease_name": "lupus_nephritis",
    "section_type": null,
    "source_type": "chunk",
    "biomarker": null
  },
  "soft_hints": {
    "disease_names": ["lupus_nephritis"],
    "section_types": ["diagnosis_criteria"],
    "biomarkers": ["protein_niệu_24h"],
    "terms": ["lupus", "ara 1997"]
  }
}
```

Nguyên tắc chính:

- Hard filter chỉ áp dụng cho request filter explicit hoặc disease từ medical tool result.
- Section/biomarker suy luận từ query được ưu tiên làm soft hint để không mất recall khi metadata chunk chưa hoàn hảo.
- Disease suy luận từ query alias cũng là soft hint, không hard filter, vì metadata disease trong corpus còn có các chunk đúng nằm dưới disease rộng.
- Nếu query mơ hồ, planner không tự gán bệnh/chỉ số cứng; nó giữ query rộng và đưa alias vào soft hints.
- `retrieval_plan.query` là input cho embedding/FTS/rerank; `retrieval_plan.filters` là hard pre-filter.
- Trong retriever, embedding được dùng với query enrich đầy đủ, nhưng full-text search chỉ dùng dòng câu hỏi gốc đầu tiên để tránh FTS bị nhiễu bởi các dòng `Bệnh trọng tâm`, `Mục cần tìm`, `Từ khóa ưu tiên`.

### Implementation update 2026-05-25 — Context expansion v3

Retriever hiện trả `content` đầy đủ hơn cho mỗi result thay vì chỉ `preview` ngắn.

Với `source_type = chunk`, retriever cũng lấy thêm chunk lân cận cùng `source_file`, `page`, `chunk_index`:

- chunk trước nếu là heading/ngữ cảnh ngắn
- chunk hiện tại
- chunk sau nếu là đoạn ngắn bổ trợ

Mục tiêu:

- Giữ heading cha cho chunk con, ví dụ `Bệnh cầu thận thay đổi tối thiểu` + `4.1.1 Lâm sàng...`
- Tăng `context_recall_required_facts_exact`
- Tăng `groundedness`
- Giảm hallucination do final prompt có đủ fact hơn

### Implementation update 2026-05-30 — Vietnamese-aware hybrid search

Retriever hiện dùng hybrid 3 nhánh:

1. Dense vector search bằng OpenAI embedding cho semantic similarity.
2. PostgreSQL full-text search `simple` cho keyword/BM25-like ranking nội bộ.
3. Lexical substring search có dấu/alias tiếng Việt cho cụm y khoa quan trọng.

Lý do thêm lexical branch:

- Dữ liệu y khoa hiện chủ yếu là tiếng Việt có dấu.
- PostgreSQL `simple` không phải tokenizer tiếng Việt, nên FTS có thể yếu với cụm như `hội chứng thận hư`, `bệnh cầu thận thay đổi tối thiểu`, `viêm cầu thận lupus`.
- Người dùng có thể gõ không dấu, có dấu, acronym hoặc alias. Embedding giúp semantic, nhưng exact phrase vẫn cần branch lexical để tăng recall.

Fusion hiện là weighted reciprocal rank fusion:

- vector weight: `0.85`
- FTS keyword weight: `1.0`
- Vietnamese lexical weight: `1.15`

Sau fusion vẫn cộng thêm:

- metadata bonus theo disease/section/biomarker hint
- lexical bonus theo token overlap, số liệu, acronym, key phrase
- neighbor context expansion để giữ heading cha và đoạn tiêu chuẩn liên quan

Chiến lược này được gọi là:

```text
Vietnamese-aware Hybrid RAG with Metadata Filtering, Weighted RRF, and Context Expansion
```

### Implementation update 2026-05-30 — Agentic retrieval refine

LangGraph có thêm node `assess_and_refine_evidence` sau `retrieve_context`.

Node này đóng vai trò evidence judge nhẹ:

- đo số evidence
- đo token coverage giữa query và context
- đo term hits từ `retrieval_plan.soft_hints`
- xem top retrieval score

Nếu evidence chưa đủ và chưa retry, graph tự tạo `agentic_retry_query` gồm:

- câu hỏi gốc
- disease/section/biomarker soft hints
- terms quan trọng
- hướng dẫn tìm đoạn giải thích trực tiếp, tiêu chuẩn, biểu hiện, phân loại hoặc điều trị

Sau đó graph search lại một lần với filter nới lỏng hơn (`source_type=chunk`, không hard disease/section/biomarker) rồi merge/dedupe evidence.

Mục tiêu:

- tăng context recall cho câu hỏi mơ hồ hoặc metadata chưa chuẩn
- giảm trả lời fallback sai khi lượt search đầu bị lọc quá chặt
- giữ chi phí thấp hơn agentic loop LLM nhiều vòng

Chiến lược này được gọi là:

```text
Deterministic Agentic RAG with Evidence Sufficiency Judge and One-shot Retrieval Retry
```

### Implementation update 2026-05-30 — Multi-query decomposition

Trước khi retrieval chính chạy xong, graph tạo thêm `agentic_queries` từ `retrieval_plan.soft_hints`.

Các sub-query được tạo theo 3 hướng:

1. Disease/section focused query: dùng bệnh trọng tâm và mục cần tìm.
2. Biomarker/term focused query: dùng chỉ số, acronym, tiêu chuẩn, cụm y khoa.
3. Multi-intent query: chỉ bật nếu câu hỏi có nhiều intent như định nghĩa + chẩn đoán + điều trị.

Ví dụ:

```text
User: Viêm cầu thận lupus được chẩn đoán khi nào theo ARA 1997?

Sub-query 1:
Viêm cầu thận lupus được chẩn đoán khi nào theo ARA 1997?
Bệnh trọng tâm: Viêm thận lupus
Mục cần tìm: Chẩn đoán

Sub-query 2:
Viêm cầu thận lupus được chẩn đoán khi nào theo ARA 1997?
Chỉ số/từ khóa: ARA 1997, protein niệu
Ưu tiên đoạn có tiêu chuẩn, ngưỡng, biểu hiện hoặc định nghĩa trực tiếp.
```

Mục tiêu:

- tăng recall cho câu hỏi dài, nhiều ý hoặc dùng thuật ngữ chuyên khoa
- tránh phụ thuộc vào một embedding query duy nhất
- vẫn không tốn LLM call vì decomposition là deterministic

### Implementation update 2026-05-30 — Evidence grading

Sau retrieval chính và sau agentic retry nếu có, graph chấm từng evidence item trước khi đưa vào final prompt.

Grade dựa trên:

- token hits giữa query và context
- term hits từ `soft_hints`
- metadata hits với hard filters
- retrieval score gốc (`fusion_score`, `keyword_score`, `similarity`)

Mỗi item được gắn:

```json
{
  "evidence_grade": {
    "score": 0.42,
    "token_hits": ["chan", "doan"],
    "term_hits": ["protein niệu"],
    "metadata_hits": 1,
    "retrieval_score": 0.13,
    "rank_before_grade": 2
  }
}
```

Sau đó graph sort lại evidence theo grade để prompt cuối nhận context liên quan nhất trước.

Mục tiêu:

- tăng `context_precision_llm`
- giảm chunk nhiễu lọt vào prompt
- giúp LangSmith debug vì `include_debug=true` sẽ thấy `evidence_grades`

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
