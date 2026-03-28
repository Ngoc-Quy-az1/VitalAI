# RAG Design

## Mục tiêu của tài liệu này

File này chốt cách làm RAG phù hợp với bộ dữ liệu hiện có.

Câu hỏi trung tâm là:

`Dữ liệu PDF hiện tại cần được chuyển thành hệ thống retrieval như thế nào để giảm sai số y khoa?`

## Kết luận trước

Không thể dùng:

- fixed-size chunking
- vector-only retrieval
- ingestion kiểu coi mọi nội dung là prose

## Vì sao RAG thông thường sẽ hỏng

Bộ dữ liệu hiện tại chứa ít nhất 4 loại nội dung khác nhau:

1. prose y khoa có cấu trúc
2. ngưỡng số liệu nằm trong câu
3. JSON nhúng thô
4. công thức và bảng phân loại

Nếu xử lý như một document text thông thường:

- numeric threshold sẽ bị mất context
- JSON block bị biến thành noise
- các bệnh thận khác nhau sẽ gây retrieval nhiễu chéo
- agent dễ trả lời đúng nghĩa nhưng sai số liệu

## Thiết kế RAG đúng cho repo này

## 1. Content classification trước chunking

Mỗi block phải được gắn một `content_type` trước khi split:

- `prose`
- `threshold_value`
- `json_block`
- `formula`
- `table_like`

Không có bước này thì các bước sau đều thiếu nền tảng.

## 2. Xử lý riêng từng loại nội dung

### Prose

- split theo heading
- giữ disease context
- không cắt giữa diagnostic criteria

### Threshold value

Ví dụ:

- `protein niệu > 3,5 g/24 giờ`
- `albumin máu < 30 g/l`
- `GFR < 60 kéo dài trên 3 tháng`

Loại này phải được extract ra structured record.

### JSON block

Phải salvage lại vì nó đã chứa:

- `text_chunks`
- `rules`
- `rows`
- metadata sơ khai

### Formula

Phải lưu structured riêng với:

- tên công thức
- biểu thức
- biến đầu vào
- đơn vị
- nguồn

## 3. Metadata-first retrieval

Semantic retrieval chỉ nên là bước thứ hai.

Bước đầu phải là xác định:

- disease
- section
- biomarker
- content type

Ví dụ:

- query về `KDIGO` không nên search trên toàn corpus
- query về `lupus nephritis` không nên bị nhiễu bởi phần CKD hay AKI

## 4. Hybrid retrieval

Sau metadata pre-filter, retrieval nên gồm:

- vector retrieval cho ngữ nghĩa
- FTS/BM25 cho keyword và acronym

Lý do:

- biomarker
- tên thuốc
- chỉ số
- stage label

thường cần exact match hơn semantic similarity.

## 5. Structured lookup song song

Đây là phần khiến VitalAI khác RAG thường.

Với lab-result input, agent không nên chỉ hỏi vector store.

Nó phải:

1. parse biomarker
2. lookup threshold structured
3. lookup formula structured nếu cần
4. dùng retrieval để lấy explanation + citation

## 6. Output contract cho ingestion

Trước khi viết parser, phải chốt 3 output:

- `chunks.jsonl`
- `thresholds.jsonl`
- `formulas.json`

Field-level contract được chốt riêng tại:

- [Metadata_Schema.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/Metadata_Schema.md)

## Metadata đề xuất

Tối thiểu cho `semantic chunks`:

```json
{
  "doc_type": "disease_guideline",
  "disease_name": "benh_than_man",
  "section_type": "classification",
  "content_type": "prose",
  "biomarker": "GFR",
  "source_file": "ten_file.pdf",
  "page": 133,
  "language": "vi"
}
```

Thêm cho `threshold_value`:

```json
{
  "threshold_op": "<",
  "threshold_value": 60,
  "threshold_unit": "ml/ph/1.73m2"
}
```

## Citation là requirement bắt buộc

Mỗi response phải có:

- source file
- page
- nếu có thể thì disease và section

Đây là safety requirement, không phải phần trang trí.

## Những gì dữ liệu hiện tại chưa đủ

Hiện dữ liệu thiên mạnh về nephrology.

Chưa đủ tốt cho:

- CBC interpretation tổng quát
- HGB/WBC/RBC/PLT normal range chuẩn
- medical assistant đa chuyên khoa

Nghĩa là:

- có thể thiết kế hệ thống ngay
- nhưng phạm vi use case cần giới hạn theo độ phủ dữ liệu

## Thứ tự triển khai đúng

1. audit tài liệu nguồn
2. chốt metadata schema
3. chốt output format của ingestion
4. chốt structured tables
5. mới code retrieval
6. rồi mới code agent graph

## Kết luận

RAG đúng cho VitalAI phải là:

- `metadata-aware`
- `disease-aware`
- `hybrid`
- `structured + semantic song song`

Nếu bỏ một trong bốn ý trên, hệ thống sẽ dễ trả lời sai ở các câu hỏi y khoa liên quan threshold.
