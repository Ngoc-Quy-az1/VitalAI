# Data Audit

## Mục tiêu của tài liệu này

File này trả lời đúng câu hỏi:

`Repo hiện đang có dữ liệu gì thật sự, và mức độ sẵn sàng của dữ liệu đó đến đâu?`

## Kết luận ngắn gọn

Repo hiện chưa có processed corpus đúng nghĩa.

Thứ đang có là:

- một PDF nephrology tương đối giá trị
- nhưng là dữ liệu thô hoặc bán-thô
- chưa được chuẩn hóa cho RAG

## Dữ liệu đã xác nhận

Hiện mới xác nhận 1 file chính:

- `data/raw_data/ĐẠI CƯƠNG VỀ BỆNH LÝ CẦU THẬN( No Table) .pdf`

Hiện trạng thư mục:

- `data/raw_data/` đang chứa file PDF nguồn
- `data/processed_data/` đang trống, chưa chứa output thật sự

Điều này hợp lý hơn so với trạng thái trước, nhưng vẫn cần chốt naming dài hạn cho rõ ràng:

- `raw_data` chứa input gốc
- `processed_data` hoặc `processed` chỉ nên chứa output đã xử lý

## Những gì có trong PDF

Qua việc đọc file, có thể tách ra 4 nhóm nội dung:

### 1. Prose y khoa có cấu trúc

Ví dụ:

- khái niệm
- phân loại
- lâm sàng
- cận lâm sàng
- chẩn đoán
- điều trị
- tiến triển
- biến chứng

### 2. Threshold nằm inline

Ví dụ:

- `protein niệu > 3,5 g/24 giờ`
- `albumin máu < 30 g/l`
- `cholesterol > 6,5 mmol/l`
- `GFR < 60 kéo dài trên 3 tháng`

Đây là lớp dữ liệu có ảnh hưởng lớn nhất đến độ an toàn câu trả lời.

### 3. JSON nhúng

PDF có các đoạn dạng:

- `text_chunks`
- `rules`
- `rows`
- metadata

Điểm quan trọng:

- đây không phải noise hoàn toàn
- đây là tài sản có thể salvage cho ingestion sau này

### 4. Công thức và bảng phân loại

Ví dụ:

- CKD stage
- KDIGO
- albumin niệu A1/A2/A3
- công thức liên quan GFR hoặc creatinin

## Domain coverage hiện tại

Bộ dữ liệu hiện tại mạnh về:

- nephrology
- CKD
- AKI
- lupus nephritis
- diabetic kidney disease
- glomerular disease

Nhưng chưa đủ tốt cho:

- chatbot lab tổng quát
- CBC interpretation
- drug safety tổng quát

## Rủi ro nếu code quá sớm

### Rủi ro 1

Chunk sai và làm mất diagnostic threshold.

### Rủi ro 2

Retrieval bị nhiễu giữa nhiều bệnh thận khác nhau.

### Rủi ro 3

Agent trả lời đúng văn cảnh nhưng sai số liệu.

## Việc phải làm trước khi code

### Bước 1 — xác nhận inventory dữ liệu

- repo còn bao nhiêu PDF nguồn
- có file Word/Excel/table nào chưa đưa vào không

### Bước 2 — chuẩn hóa naming

Đề xuất:

- `data/raw/` hoặc `data/raw_data/` cho file nguồn
- `data/processed/` cho output thật sự

### Bước 3 — chốt taxonomy tài liệu

Mỗi nguồn nên được gắn một nhóm:

- guideline
- threshold/reference range
- formula/classification
- medication

### Bước 4 — chốt output của ingestion

Tối thiểu nên có:

- chunk JSONL
- threshold JSONL
- formula JSON

## Kết luận

Việc đúng tiếp theo không phải là code agent.

Việc đúng là:

1. chốt data inventory
2. chốt taxonomy
3. chốt output format

Sau ba việc này mới nên bắt đầu viết ingestion.
