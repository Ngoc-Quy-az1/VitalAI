# Database Design

## Mục tiêu của tài liệu này

File này mô tả **logical database design** cho VitalAI.

Đây chưa phải schema chính thức để implement ngay. Nó dùng để:

- xác định loại dữ liệu nào sẽ được lưu ở đâu
- tránh nhét toàn bộ tri thức y khoa vào một vector table

Field-level contract cho output xử lí data được mô tả riêng tại:

- [Metadata_Schema.md](c:/Users/ngtru/Documents/CodeSpace/VitalAI/doc/Metadata_Schema.md)

## Nguyên tắc trung tâm

VitalAI phải có **2 lớp lưu trữ tri thức**:

1. `semantic store`
2. `structured store`

Nếu chỉ có semantic store thì hệ thống sẽ yếu ở:

- threshold lookup
- numeric comparison
- formula execution
- staging rules

## Các bảng logic nên có

### 1. `medical_documents`

Dùng cho:

- prose chunks
- explanatory chunks
- disease guideline
- text được salvage từ JSON nhúng

Đây là bảng phục vụ retrieval ngữ nghĩa.

Nên có:

- content
- embedding
- metadata
- created_at

## 2. `medical_thresholds`

Dùng cho:

- ngưỡng chẩn đoán
- ngưỡng phân loại
- normal range
- stage rule dạng exact value

Đây là bảng phục vụ lookup chính xác.

Nên có:

- biomarker
- threshold_op
- threshold_value
- threshold_unit
- disease_name
- section_type
- source_file
- page

## 3. `medical_formulas`

Dùng cho:

- công thức
- biến đầu vào
- đơn vị kết quả
- giải thích ngắn

Nên có:

- formula_name
- formula_expr
- variables
- output_unit
- source_file
- page

## 4. `patient_sessions`

Dùng cho:

- history
- extracted lab values
- multi-turn follow-up

Giai đoạn đầu có thể chưa cần implement ngay, nhưng nên giữ trong thiết kế.

## Metadata quan trọng nhất

Với `medical_documents`, cần ưu tiên:

- `doc_type`
- `disease_name`
- `section_type`
- `content_type`
- `biomarker`
- `source_file`
- `page`
- `language`

Đây là nền tảng cho metadata pre-filter.

## Khi nào dùng semantic store

Dùng semantic store cho:

- giải thích bệnh
- treatment prose
- clinical context
- đoạn văn dài cần hiểu nghĩa

## Khi nào phải dùng structured store

Không nên chỉ dựa vào vector cho:

- `GFR < 60`
- `ACR > 300`
- `protein niệu > 3.5 g/24h`
- `KDIGO G3b`
- công thức

Các loại này phải có structured table riêng.

## Query logic mong muốn

### Với question input

- metadata pre-filter trước
- semantic/hybrid retrieval sau

### Với lab input

- parse biomarker trước
- lookup threshold/formula structured trước
- semantic retrieval chỉ để bổ sung explanation và citation

## Mapping từ output file sang logical tables

### `chunks.jsonl`

Ánh xạ vào:

- `medical_documents`

### `thresholds.jsonl`

Ánh xạ vào:

- `medical_thresholds`

### `formulas.json`

Ánh xạ vào:

- `medical_formulas`

## Thứ tự chốt thiết kế DB

1. chốt taxonomy tài liệu
2. chốt metadata schema
3. chốt structured fields cho threshold/formula
4. rồi mới viết schema SQL

## Kết luận

DB đúng cho VitalAI không phải là một vector table duy nhất.

Nó phải là:

- `documents` cho ngữ nghĩa
- `thresholds` cho exact rules
- `formulas` cho tính toán
- `sessions` cho hội thoại
