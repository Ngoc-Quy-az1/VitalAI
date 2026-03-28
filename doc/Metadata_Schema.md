# Metadata Schema

## Mục tiêu của tài liệu này

File này chốt `data contract` cho output của phase xử lí data.

Nó trả lời 3 câu hỏi:

1. `chunk` sẽ có những field nào
2. `threshold` sẽ có những field nào
3. `formula` sẽ có những field nào

Đây là bước trung gian bắt buộc giữa:

- `data audit`
- `ingestion spec`

Nếu chưa chốt file này, việc viết ingestion sẽ dễ lệch hướng.

## Nguyên tắc thiết kế schema

Schema phải phục vụ đồng thời 3 nhu cầu:

1. retrieval
2. exact lookup
3. citation

Do đó mỗi record không chỉ cần nội dung, mà còn cần:

- nguồn gốc
- loại nội dung
- disease context
- section context

## Output 1 — `chunks.jsonl`

Đây là output cho semantic retrieval.

Mỗi dòng là một chunk.

### Schema đề xuất

```json
{
  "chunk_id": "ckd_p133_001",
  "content": "Mức lọc cầu thận (GFR) < 60 ml/ph/1,73m2 kéo dài trên 3 tháng là tiêu chuẩn chẩn đoán bệnh thận mạn.",
  "metadata": {
    "doc_type": "disease_guideline",
    "disease_name": "benh_than_man",
    "section_type": "definition",
    "content_type": "prose",
    "biomarker": "GFR",
    "source_file": "ĐẠI CƯƠNG VỀ BỆNH LÝ CẦU THẬN( No Table) .pdf",
    "page": 133,
    "language": "vi",
    "chunk_index": 1
  }
}
```

### Field bắt buộc

- `chunk_id`
- `content`
- `metadata.doc_type`
- `metadata.disease_name`
- `metadata.section_type`
- `metadata.content_type`
- `metadata.source_file`
- `metadata.page`
- `metadata.language`
- `metadata.chunk_index`

### Field optional nhưng nên có

- `metadata.biomarker`
- `metadata.formula_name`
- `metadata.parent_heading`
- `metadata.source_block_type`

### Giá trị gợi ý cho `doc_type`

- `disease_guideline`
- `threshold_reference`
- `formula_reference`
- `medication_reference`

### Giá trị gợi ý cho `content_type`

- `prose`
- `json_block`
- `table_like`
- `formula_explanation`
- `classification_explanation`

### Giá trị gợi ý cho `section_type`

- `definition`
- `classification`
- `clinical_features`
- `diagnosis_criteria`
- `treatment`
- `progression`
- `complications`
- `follow_up`

## Output 2 — `thresholds.jsonl`

Đây là output cho exact numeric lookup.

Mỗi dòng là một threshold hoặc một rule định lượng.

### Schema đề xuất

```json
{
  "threshold_id": "ckd_gfr_lt_60_p133",
  "biomarker": "GFR",
  "threshold_op": "<",
  "threshold_value": 60,
  "threshold_unit": "ml/ph/1.73m2",
  "label": "benh_than_man_definition",
  "severity": "high",
  "disease_name": "benh_than_man",
  "section_type": "definition",
  "content_type": "threshold_value",
  "source_text": "Mức lọc cầu thận (GFR) < 60 ml/ph/1,73m2 kéo dài trên 3 tháng là tiêu chuẩn chẩn đoán bệnh thận mạn.",
  "source_file": "ĐẠI CƯƠNG VỀ BỆNH LÝ CẦU THẬN( No Table) .pdf",
  "page": 133,
  "language": "vi"
}
```

### Field bắt buộc

- `threshold_id`
- `biomarker`
- `threshold_op`
- `threshold_value`
- `threshold_unit`
- `disease_name`
- `section_type`
- `content_type`
- `source_text`
- `source_file`
- `page`
- `language`

### Field optional nhưng quan trọng

- `label`
- `severity`
- `normal_range_group`
- `applies_to_sex`
- `applies_to_age_group`
- `duration_condition`
- `notes`

### Giá trị gợi ý cho `threshold_op`

- `<`
- `<=`
- `>`
- `>=`
- `between`

Nếu là `between`, nên thêm:

- `threshold_value_min`
- `threshold_value_max`

### Gợi ý chuẩn hóa biomarker

Nên dùng tên canonical nhất quán:

- `GFR`
- `ACR`
- `protein_niệu_24h`
- `albumin_máu`
- `cholesterol`

## Output 3 — `formulas.json`

Đây là output cho formula engine và rule engine.

Mỗi object là một công thức hoặc rule phân loại có thể áp dụng.

### Schema đề xuất

```json
{
  "formula_id": "mdrd_gfr",
  "formula_name": "MDRD",
  "formula_type": "calculation",
  "expression": "175 * (creatinine_mg_dl ^ -1.154) * (age ^ -0.203) * sex_factor",
  "variables": [
    {
      "name": "creatinine_mg_dl",
      "description": "Creatinin huyết thanh",
      "unit": "mg/dL",
      "required": true
    },
    {
      "name": "age",
      "description": "Tuổi bệnh nhân",
      "unit": "years",
      "required": true
    }
  ],
  "output_name": "GFR",
  "output_unit": "ml/ph/1.73m2",
  "disease_name": "benh_than_man",
  "section_type": "classification",
  "source_text": "GFR có thể được ước tính bằng công thức MDRD dựa trên creatinin huyết thanh, tuổi, giới và chủng tộc.",
  "source_file": "ĐẠI CƯƠNG VỀ BỆNH LÝ CẦU THẬN( No Table) .pdf",
  "page": 133,
  "language": "vi"
}
```

### Field bắt buộc

- `formula_id`
- `formula_name`
- `formula_type`
- `expression`
- `variables`
- `output_name`
- `output_unit`
- `disease_name`
- `section_type`
- `source_text`
- `source_file`
- `page`
- `language`

### Field optional nhưng nên có

- `interpretation`
- `normal_range`
- `applicability`
- `assumptions`
- `notes`

### Giá trị gợi ý cho `formula_type`

- `calculation`
- `classification_rule`
- `staging_rule`

## Mapping giữa 3 output

Ba output này không độc lập hoàn toàn.

### `chunks.jsonl`

Dùng cho:

- retrieval ngữ nghĩa
- citation
- explanation

### `thresholds.jsonl`

Dùng cho:

- exact lookup
- compare value
- flag abnormality

### `formulas.json`

Dùng cho:

- calculate derived value
- classify stage
- explain formula source

## Rule đặt ID

ID nên deterministic để dễ trace.

### Gợi ý

- `chunk_id`: `{disease}_{page}_{chunk_index}`
- `threshold_id`: `{disease}_{biomarker}_{op}_{value}_p{page}`
- `formula_id`: tên canonical của formula hoặc rule

## Mức tối thiểu phải chốt trước khi code ingestion

Phải thống nhất được:

1. danh sách `doc_type`
2. danh sách `section_type`
3. danh sách `content_type`
4. canonical biomarker names
5. schema cho 3 output trên

## Kết luận

Sau file này, phase tiếp theo có thể làm được là:

- viết ingestion spec

Chưa cần code ngay, nhưng từ đây việc viết code đã có contract rõ ràng.
