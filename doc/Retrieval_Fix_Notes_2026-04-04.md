# Retrieval Fix Notes — 2026-04-04

## Mục tiêu của tài liệu này

File này ghi lại toàn bộ các thay đổi đã được thực hiện để xử lý vấn đề retrieval bị lệch ngữ cảnh, đặc biệt với case:

- query: `lupus ban đỏ là gì`
- expected chunk: `lupus_nephritis_p25_002`
- kết quả cũ hay lệch sang:
  - `lupus_nephritis_p27_004`
  - `lupus_nephritis_p28_001`

Tài liệu này tập trung vào 3 câu hỏi:

1. Đã thay đổi những gì?
2. Mỗi thay đổi có tác dụng gì?
3. Quy trình regenerate dữ liệu và kiểm tra lại hoạt động như thế nào?

## Bối cảnh lỗi

Qua review code, docs và artifacts hiện tại, lỗi không nằm ở một chỗ đơn lẻ mà là tổ hợp của nhiều nguyên nhân:

### 1. Retrieval đang là vector-only

Retriever cũ trong `src/LLM/retrieval/vector_search.py` chỉ làm:

1. embed query
2. cosine search top-k
3. trả kết quả

Trong khi `doc/Retrieval_Spec.md` của dự án đã mô tả hướng đúng là:

1. hiểu query
2. suy ra intent metadata
3. pre-filter / hint
4. hybrid retrieval
5. rank fusion

### 2. Chunk định nghĩa bị quá dài và bị loãng nghĩa

Chunk `lupus_nephritis_p25_002` trước khi sửa chứa quá nhiều ý trong cùng một block:

- định nghĩa
- tiên lượng
- tầm soát
- tổng quan viêm thận lupus
- đoạn nói về điều trị/phân loại mô bệnh học

Khi một chunk quá dài và trộn nhiều ý, embedding dễ bị loãng, nên query kiểu `là gì` có thể không kéo đúng phần định nghĩa lên top.

### 3. Metadata còn bị nhiễu

Một số metadata trước khi sửa bị sai hoặc chưa đủ tốt cho retrieval:

- `lupus_nephritis_p25_002` từng bị gán `doc_type = medication_reference`
- `lupus_nephritis_p27_004` từng bị gán `section_type = clinical_features` dù thực chất là phần chẩn đoán
- một số chunk khác bị false-positive biomarker như `sodium`
- nhiều section bị gán theo từ khóa xuất hiện sâu trong thân đoạn thay vì heading

## Các file đã thay đổi

### Code

- `src/LLM/ingestion/processor.py`
- `src/LLM/retrieval/vector_search.py`

### Artifacts được regenerate

- `data/processed_data/chunks.jsonl`
- `data/processed_data/thresholds.jsonl`
- `data/processed_data/formulas.json`
- `data/processed_data/summary.json`
- `data/embedding_data/embedding_documents.jsonl`
- `data/embedding_data/embedding_manifest.json`

## Thay đổi chi tiết

## A. Sửa ingestion / chunking

File chính:

- `src/LLM/ingestion/processor.py`

### A1. Tách prose block dài theo bullet nhưng vẫn giữ heading

Đã thêm logic:

- `_split_large_prose_block(...)`
- `_split_block_units(...)`
- `_compose_prose_subchunk(...)`
- `_is_bullet_like_line(...)`

### Tác dụng

- giảm tình trạng một chunk ôm quá nhiều ý
- giữ heading như `1. KHÁI NIỆM` ở đầu các subchunk để semantic signal vẫn rõ
- tăng khả năng query definition match đúng chunk định nghĩa

### Hiệu quả trên case lupus

Trang 25 trước đây có một block rất dài.

Sau khi sửa, phần `KHÁI NIỆM` được tách thành nhiều chunk nhỏ hơn:

- `lupus_nephritis_p25_002`
- `lupus_nephritis_p25_003`
- `lupus_nephritis_p25_004`
- `lupus_nephritis_p25_005`

Trong đó `lupus_nephritis_p25_002` giờ giữ phần định nghĩa cốt lõi hơn, không còn bị loãng như trước.

## B. Sửa metadata detection

File chính:

- `src/LLM/ingestion/processor.py`

### B1. Tính `section_type` và `biomarker` trước, rồi dùng lại khi build metadata

Đã đổi `_build_metadata(...)` để:

- detect `section_type`
- detect `biomarker`
- truyền lại vào `_detect_doc_type(...)`

### Tác dụng

- tránh chuyện mỗi field detect một lần riêng dẫn đến lệch logic
- giúp `doc_type` bám sát ngữ cảnh section thực hơn

### B2. Siết lại heuristic `doc_type`

Đã sửa `_detect_doc_type(...)` theo hướng:

- chỉ gán `medication_reference` khi đoạn thực sự thuộc `treatment` và có tín hiệu thuốc/liều dùng
- chỉ gán `threshold_reference` khi có biomarker và đoạn thuộc `classification` hoặc `diagnosis_criteria`
- các trường hợp còn lại ưu tiên `disease_guideline`

### Tác dụng

- tránh việc đoạn định nghĩa bị gán nhầm thành tài liệu điều trị
- giảm metadata noise cho retrieval layer

### B3. Ưu tiên heading và phần đầu đoạn khi detect `section_type`

Đã sửa `_detect_section_type(...)` để:

- nhìn `heading_preview` trước
- chỉ fallback sang một `section_preview` ngắn ở đầu block
- không còn scan quá rộng toàn body như trước

### Tác dụng

- giảm false-positive do các từ khóa như `điều trị`, `chẩn đoán`, `mô bệnh học` xuất hiện sâu trong thân đoạn
- giữ `section_type` phản ánh đúng heading logic của tài liệu

### Hiệu quả quan sát được

Sau khi regenerate:

- `lupus_nephritis_p25_002` là `section_type = definition`
- `lupus_nephritis_p25_002` là `doc_type = disease_guideline`
- `lupus_nephritis_p27_004` là `section_type = diagnosis_criteria`
- `lupus_nephritis_p26_001` không còn bị kéo sang `treatment`, mà về `general`

### B4. Giảm false-positive biomarker

Đã bỏ alias:

- `sodium: ["na "]`

Giữ lại:

- `na+`
- `natri`

### Tác dụng

- tránh match nhầm nhiều đoạn bình thường chỉ vì có chuỗi ký tự `na`
- giảm metadata biomarker sai

## C. Nâng retrieval từ vector-only sang hybrid

File chính:

- `src/LLM/retrieval/vector_search.py`

### C1. Thêm query understanding ở mức heuristic

Đã thêm các map:

- `DISEASE_HINTS`
- `SECTION_HINTS`
- `BIOMARKER_HINTS`

Và các hàm:

- `_understand_query(...)`
- `_detect_hint(...)`
- `_contains_keyword(...)`
- `_normalize_ascii(...)`

### Tác dụng

- suy ra bệnh trọng tâm
- suy ra section cần tìm
- suy ra biomarker cần ưu tiên

### Ví dụ

Query:

- `lupus ban đỏ là gì`

Được hiểu thành:

- `disease_hint = lupus_nephritis`
- `section_hint = definition`

Query:

- `ACR bao nhiêu thì là A3?`

Được hiểu thành:

- `biomarker_hint = ACR`
- `section_hint = classification`

### C2. Enrich embedding query theo intent

Thay vì embed raw query thuần túy, retriever giờ tạo query giàu ngữ cảnh hơn, ví dụ:

```text
Câu hỏi người dùng: lupus ban đỏ là gì
Bệnh trọng tâm: Viêm thận lupus
Mục cần tìm: Khái niệm
```

### Tác dụng

- giúp vector search hiểu rõ đây là câu hỏi định nghĩa chứ không phải câu hỏi chẩn đoán

### C3. Thêm full-text retrieval

Đã thêm `_search_keyword_rows(...)` dùng:

- `websearch_to_tsquery('simple', ...)`
- `to_tsvector('simple', content)`
- `ts_rank_cd(...)`

### Tác dụng

- giữ được các chunk có exact phrase hoặc keyword mạnh
- hỗ trợ tốt hơn cho query:
  - `là gì`
  - acronym
  - stage/rule
  - disease name

### C4. Hybrid fusion bằng RRF + metadata bonus

Đã thêm:

- `_fuse_rows(...)`
- `_metadata_bonus(...)`

Logic hiện tại:

1. lấy candidate từ vector retrieval
2. lấy candidate từ FTS retrieval
3. cộng điểm RRF cho cả hai nguồn
4. cộng bonus nếu match:
   - `disease_hint`
   - `section_hint`
   - `biomarker_hint`

### Tác dụng

- không phụ thuộc hoàn toàn vào vector similarity
- ưu tiên đúng chunk definition khi query là `là gì`
- vẫn giữ khả năng match semantic khi user viết query tự nhiên

## Kết quả cụ thể trên case lupus

### Trước khi sửa

Case:

- query: `lupus ban đỏ là gì`

Vấn đề:

- retriever có xu hướng kéo các chunk chẩn đoán lên:
  - `lupus_nephritis_p27_004`
  - `lupus_nephritis_p28_001`

### Sau khi sửa dữ liệu

Chunk đích hiện có metadata tốt hơn:

- `lupus_nephritis_p25_002`
  - `section_type = definition`
  - `doc_type = disease_guideline`
  - nội dung ngắn gọn hơn, tập trung hơn

Hai chunk nhiễu chính:

- `lupus_nephritis_p27_004`
  - `section_type = diagnosis_criteria`
- `lupus_nephritis_p28_001`
  - `section_type = diagnosis_criteria`

### Smoke check cục bộ

Một phép kiểm tra lexical đơn giản trên 3 chunk đích/nhiễu cho query đã enrich:

- `chunk::lupus_nephritis_p25_002` -> score `23`
- `chunk::lupus_nephritis_p27_004` -> score `17`
- `chunk::lupus_nephritis_p28_001` -> score `6`

Điều này cho thấy hướng rank mới đã nghiêng đúng về chunk định nghĩa.

## Quy trình đã thực hiện

## Bước 1. Đọc docs và xác nhận expected architecture

Đã đọc:

- `README.md`
- `doc/Ingestion_Spec.md`
- `doc/Retrieval_Spec.md`

Kết luận:

- docs của dự án đã định nghĩa retrieval theo hướng hybrid/query-aware
- code cũ chưa đi theo đúng contract đó

## Bước 2. Kiểm tra artifacts hiện tại

Đã đối chiếu:

- `data/processed_data/chunks.jsonl`
- `data/embedding_data/embedding_documents.jsonl`

Để so:

- chunk đúng
- chunk bị rank sai
- metadata của từng chunk

## Bước 3. Sửa code ingestion

Đã sửa:

- chunking
- section detection
- doc type detection
- biomarker alias

## Bước 4. Sửa code retrieval

Đã thêm:

- query understanding
- enriched query embedding
- full-text search
- fusion + metadata bonus

## Bước 5. Regenerate processed data

Đã chạy bằng virtualenv của dự án:

```bash
../.venv/bin/python scripts/process_medical_data.py
```

## Bước 6. Regenerate embedding-ready artifacts

Đã chạy:

```bash
../.venv/bin/python scripts/prepare_embedding_data.py
```

## Bước 7. Smoke test cục bộ

Đã kiểm tra:

- metadata mới của các chunk lupus
- query understanding
- lexical match giữa query enrich và 3 chunk đang quan tâm

## Cách chạy lại trên máy / môi trường thật

## 1. Regenerate processed data

```bash
../.venv/bin/python scripts/process_medical_data.py
```

## 2. Regenerate embedding documents

```bash
../.venv/bin/python scripts/prepare_embedding_data.py
```

## 3. Re-index lên Neon

Nếu muốn thay đổi có hiệu lực trên DB retrieval thật, cần re-index:

```bash
../.venv/bin/python scripts/embed_and_index.py
```

Yêu cầu:

- có `OPENAI_API_KEY`
- có `NEON_DATABASE_URL`
- các biến trong `.env` đã đúng

## 4. Test retrieval thực tế

Ví dụ:

```bash
../.venv/bin/python scripts/test_retrieval.py --query "lupus ban đỏ là gì" --top-k 5
```

Hoặc:

```bash
../.venv/bin/python scripts/test_retrieval.py --query "lupus ban đỏ là gì" --disease-name lupus_nephritis --top-k 5
```

## Giới hạn còn lại

Dù đã cải thiện đáng kể, hiện vẫn còn các giới hạn:

### 1. Query understanding vẫn là heuristic

Chưa có:

- classifier bằng model
- query rewriting thật sự
- normalization theo ontology bệnh/chỉ số

### 2. Hybrid hiện mới là vector + PostgreSQL FTS

Chưa có:

- BM25 riêng
- learned reranker
- structured lookup layer riêng

### 3. Metadata của dữ liệu nguồn vẫn còn khả năng nhiễu ở các phần khác

Đặc biệt vì PDF gốc trộn nhiều kiểu nội dung:

- prose
- threshold
- classification
- nội dung OCR nhiễu

Nên một số section/doc_type khác vẫn có thể cần tuning thêm sau khi test nhiều query hơn.

## Đề xuất bước tiếp theo

1. Re-index lại Neon bằng artifact mới.
2. Chạy lại test query thật cho các case:
   - `lupus ban đỏ là gì`
   - `điều trị lupus nephritis`
   - `ACR bao nhiêu thì là A3?`
3. Ghi lại top-k kết quả thực tế sau re-index.
4. Nếu vẫn còn lệch rank, bước tiếp theo nên là:
   - thêm structured lookup cho threshold/formula
   - thêm reranker nhẹ ở lớp cuối
   - bổ sung test set retrieval regression

## Tóm tắt ngắn

Các thay đổi vừa làm có tác dụng chính là:

1. làm chunk định nghĩa bớt loãng
2. làm metadata đúng ngữ cảnh hơn
3. làm retriever hiểu rằng `là gì` thường là query `definition`
4. giảm khả năng chunk chẩn đoán lấn át chunk định nghĩa trong case lupus

## Phần bổ sung: tích hợp LLM bằng Mistral

Sau khi retrieval đã được sửa, hệ thống đã được nối thêm một lớp answer synthesis dùng Mistral làm model chính.

### File mới / file thay đổi

- `src/LLM/qa/__init__.py`
- `src/LLM/qa/answering.py`
- `scripts/test_answer.py`
- `test_query.sh`
- `requirements.txt`

### Model chính đang dùng

Answer layer dùng:

```python
from langchain_mistralai import ChatMistralAI
```

Các biến môi trường đang được đọc:

- `MISTRAL_CLIENT_API_KEY`
- `MODEL_NAME`
- `MISTRAL_TEMPERATURE` (optional, default hiện là `0.1`)

### Luồng hoạt động mới

Hiện tại full flow QA hoạt động theo thứ tự:

1. nhận query người dùng
2. chạy hybrid retrieval trên Neon qua `NeonVectorSearcher`
3. lấy top evidence
4. format evidence thành prompt ngắn, có `source_id`, `page`, `section`
5. gọi `ChatMistralAI`
6. trả về:
   - `answer`
   - `query_understanding`
   - `results`

### Tác dụng của lớp LLM mới

- hệ thống không chỉ trả top-k chunks nữa
- có thể trả lời trực tiếp bằng tiếng Việt
- vẫn giữ retrieval debug info để dễ audit
- prompt buộc model:
  - chỉ dùng evidence đã truy xuất
  - nêu rõ nếu thiếu dữ kiện
  - thêm citation ngắn theo `source_id` và trang

### Kết quả smoke test

Đã smoke test thành công việc khởi tạo:

- `RetrievalAugmentedAnswerer`
- `ChatMistralAI`
- `NeonVectorSearcher`

Đã chạy được end-to-end query:

- `Lupus ban đỏ là gì?`

Kết quả trả lời đã bám đúng evidence chính từ:

- `lupus_nephritis_p25_002`

### Lệnh test mới

```bash
../.venv/bin/python scripts/test_answer.py --query "Lupus ban đỏ là gì?" --top-k 5
```

### Ghi chú vận hành

- Nếu retrieval DB chưa được re-index sau các thay đổi ingestion/embedding, answer flow vẫn chạy nhưng có thể còn dùng metadata cũ ở một số record.
- Vì vậy sau khi thay đổi artifacts, cần chạy lại:

```bash
../.venv/bin/python scripts/embed_and_index.py
```
