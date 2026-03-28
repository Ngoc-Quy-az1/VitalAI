# Ingestion Spec

## Mục tiêu của tài liệu này

File này mô tả chính xác cách xử lí tài liệu nguồn hiện có để tạo ra 3 output đã chốt:

- `chunks.jsonl`
- `thresholds.jsonl`
- `formulas.json`

Đây là tài liệu cầu nối giữa:

- `Metadata_Schema.md`
- giai đoạn bắt đầu viết code ingestion

## Phạm vi áp dụng hiện tại

Spec này được viết cho đúng dữ liệu hiện đang có:

- `data/raw_data/ĐẠI CƯƠNG VỀ BỆNH LÝ CẦU THẬN( No Table) .pdf`

Vì hiện chỉ có một tài liệu tổng hợp, ingestion pipeline giai đoạn đầu nên được tối ưu riêng cho dạng tài liệu này thay vì cố làm generic quá sớm.

## Mục tiêu của ingestion

Biến một PDF hỗn hợp thành ba lớp dữ liệu:

1. `semantic chunks`
   Dùng cho retrieval ngữ nghĩa.

2. `threshold records`
   Dùng cho exact lookup và so sánh số liệu.

3. `formula / classification records`
   Dùng cho tính toán và staging.

## Nguyên tắc chung

### 1. Không chunk trước khi phân loại block

Pipeline không được bắt đầu bằng:

- cắt 512 token
- rồi mới đoán block là gì

Phải làm ngược lại:

1. extract text theo page
2. detect block boundary
3. classify block
4. route block vào output phù hợp

### 2. Ưu tiên giữ nguyên ngữ cảnh bệnh

Nếu một đoạn thuộc:

- cùng một bệnh
- cùng một section
- cùng một tiêu chuẩn chẩn đoán

thì không nên cắt rời vô tội vạ.

### 3. Numeric rule phải được tách khỏi prose

Bất kỳ câu nào chứa threshold lâm sàng phải được đi qua nhánh trích xuất structured.

## Pipeline tổng thể

```text
PDF source
  |
  v
Page text extraction
  |
  v
Block segmentation
  |
  v
Content classification
  |
  +--> prose
  |      -> semantic chunking
  |      -> chunks.jsonl
  |
  +--> threshold_value
  |      -> normalize biomarker/operator/unit
  |      -> thresholds.jsonl
  |      -> optional companion chunk
  |
  +--> json_block
  |      -> salvage text_chunks / rules / rows
  |      -> route to chunks / thresholds / formulas
  |
  +--> formula_or_classification
         -> formulas.json
         -> optional companion chunk
```

## Bước 1 — Page text extraction

### Input

- PDF file

### Output

Danh sách page-level text:

```json
{
  "page": 133,
  "text": "..."
}
```

### Yêu cầu

- giữ nguyên page number
- không normalize mất dấu tiếng Việt
- giữ line breaks ở mức đủ để nhận diện heading

### Không làm ở bước này

- không chunk
- không extract threshold
- không parse formula

## Bước 2 — Block segmentation

## Mục tiêu

Chia text theo block logic thay vì token count.

### Block boundary ưu tiên

1. heading kiểu số:
- `1. KHÁI NIỆM`
- `4.1.1. Lâm sàng và cận lâm sàng`
- `9.2.1. Điều trị`

2. block JSON:
- bắt đầu từ `{`
- chứa key như `text_chunks`, `rules`, `rows`, `formula`

3. block bảng text hóa:
- có cấu trúc lặp cột/row rõ ràng
- hoặc có field dạng `name`, `dose`, `stage`, `range`

### Output

Danh sách block:

```json
{
  "page": 133,
  "block_index": 4,
  "raw_text": "...",
  "block_hint": "json_like"
}
```

## Bước 3 — Content classification

Mỗi block phải được gắn đúng `content_type`.

### Các loại cần hỗ trợ

- `prose`
- `threshold_value`
- `json_block`
- `formula`
- `classification_rule`
- `table_like`

### Rule nhận diện gợi ý

#### `prose`

- có câu dài
- có heading/paragraph
- không có cấu trúc key-value rõ

#### `threshold_value`

- có biomarker + toán tử + giá trị + đơn vị
- ví dụ:
  - `> 3,5 g/24 giờ`
  - `< 30 g/l`
  - `> 300 mg/g`

#### `json_block`

- chứa `{`, `}`, `[]`
- có key như:
  - `text_chunks`
  - `rules`
  - `rows`
  - `metadata`

#### `formula`

- có công thức tính
- có tên công thức
- có input/output hoặc mô tả biến

#### `classification_rule`

- có các stage/rule như:
  - `G1 -> G5`
  - `A1 -> A3`
  - `if ... then ...`

#### `table_like`

- dữ liệu liệt kê row
- thường dùng cho thuốc, stage, nguyên nhân-cơ chế

## Bước 4 — Routing theo loại block

## A. Prose -> `chunks.jsonl`

### Cách xử lí

- split tiếp theo heading con nếu block quá dài
- giữ disease context
- giữ section context

### Điều kiện split

Chỉ split thêm nếu:

- block quá dài
- chứa nhiều subsection khác nhau

Không split nếu:

- block đang là một tiêu chuẩn chẩn đoán liên tục
- split sẽ làm mất quan hệ giữa disease và threshold

### Output

- 1 hoặc nhiều record trong `chunks.jsonl`

## B. Threshold block -> `thresholds.jsonl`

### Cách xử lí

Từ câu gốc, trích:

- biomarker
- threshold_op
- threshold_value
- threshold_unit
- disease_name
- section_type
- source_text

### Ví dụ

Từ:

`Mức lọc cầu thận (GFR) < 60 ml/ph/1,73m2 kéo dài trên 3 tháng là tiêu chuẩn chẩn đoán bệnh thận mạn.`

Trích ra:

- biomarker: `GFR`
- threshold_op: `<`
- threshold_value: `60`
- threshold_unit: `ml/ph/1.73m2`
- duration_condition: `> 3 months`

### Có tạo companion chunk không

Có thể có.

Nếu threshold nằm trong một câu giải thích giàu ngữ nghĩa, nên:

- tạo 1 threshold record
- đồng thời giữ 1 semantic chunk companion

## C. Formula block -> `formulas.json`

### Cách xử lí

Trích:

- formula name
- expression
- variables
- output name
- output unit
- source

### Có tạo companion chunk không

Có thể có nếu:

- block giải thích ý nghĩa lâm sàng của công thức

## D. Classification rule -> `formulas.json` hoặc `thresholds.jsonl`

### Nguyên tắc

Không phải mọi rule đều là formula.

#### Nếu là rule kiểu exact stage threshold

Ví dụ:

- `ACR < 30 -> A1`
- `30 <= ACR <= 300 -> A2`

thì nên đi vào `thresholds.jsonl`.

#### Nếu là rule phức tạp hơn hoặc cần execution logic

thì có thể đi vào `formulas.json` dưới dạng:

- `formula_type = classification_rule`

## E. JSON block -> salvage

### Ưu tiên parse những phần sau

1. `text_chunks`
   -> route vào `chunks.jsonl`

2. `rules`
   -> route vào `thresholds.jsonl` hoặc `formulas.json`

3. `rows`
   -> tùy ngữ nghĩa, route vào:
   - `thresholds.jsonl`
   - `formulas.json`
   - hoặc chunk companion

### Không nên làm

- không nhét nguyên khối JSON vào một chunk duy nhất nếu bên trong đã có cấu trúc tốt hơn

## Bước 5 — Metadata enrichment

Sau khi route xong, mỗi record phải được enrich metadata.

### Field bắt buộc phải suy ra được

- `doc_type`
- `disease_name`
- `section_type`
- `content_type`
- `source_file`
- `page`
- `language`

### Field nên enrich nếu có thể

- `biomarker`
- `formula_name`
- `parent_heading`
- `severity`

## Bước 6 — Validation trước khi ghi output

### Rule cho `chunks.jsonl`

Reject nếu:

- không có `content`
- không có `source_file`
- không có `page`
- không có `content_type`

### Rule cho `thresholds.jsonl`

Reject nếu:

- không parse được `biomarker`
- không parse được `threshold_op`
- không parse được `threshold_value`
- không có `source_text`

### Rule cho `formulas.json`

Reject nếu:

- không có `formula_name`
- không có `expression` hoặc rule logic tương đương
- không xác định được `source_file` và `page`

## Manual review bucket

Một số block không nên tự động ép vào output chính.

Nên có bucket review thủ công cho:

- OCR lỗi nặng
- block vừa có prose vừa có JSON nhưng parse không chắc
- công thức bị mất biến
- threshold không xác định được biomarker rõ ràng

## Chỉ số chất lượng cho ingestion

Sau khi viết code ingestion, nên kiểm theo các chỉ số sau:

### Coverage

- bao nhiêu page có block được parse
- bao nhiêu JSON block được salvage

### Structured extraction quality

- số threshold record hợp lệ
- số formula/classification record hợp lệ

### Semantic chunk quality

- chunk có giữ đúng disease context không
- chunk có bị cắt giữa diagnostic criteria không

## Kết quả mong muốn sau phase này

Sau ingestion phase, repo nên có:

- `data/processed/chunks.jsonl`
- `data/processed/thresholds.jsonl`
- `data/processed/formulas.json`
- 1 file summary thống kê số lượng record theo loại

## Điều chưa làm ở phase này

Chưa làm:

- embedding
- vector indexing
- retrieval
- LangGraph
- API

## Kết luận

Sau file này, bước tiếp theo hợp lý là:

- viết `Retrieval_Spec.md`

Vì đến đây:

- input data đã rõ
- output data đã rõ
- cách route block đã rõ

Nghĩa là có thể bắt đầu thiết kế retrieval trên một data contract ổn định.
