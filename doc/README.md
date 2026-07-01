# VitalAI Medical Tools Service

Service này tách riêng phần xử lý structured medical knowledge khỏi AI/RAG service.

Mục tiêu:

- Đọc `data/processed_data/thresholds.jsonl` để so ngưỡng và phân loại chỉ số.
- Đọc thêm `data/processed_data/thresholds_extra.jsonl` nếu có để bổ sung numeric rules trích từ prose chunks.
- Đọc `data/processed_data/formulas.json` để tính công thức y khoa bằng evaluator an toàn.
- Expose FastAPI endpoint để MCP adapter hoặc AI service gọi qua HTTP.
- Deploy độc lập với chatbot service.

## Cấu trúc

- `app.py`: FastAPI app và HTTP endpoints.
- `service.py`: engine parse input, lookup threshold, calculate formula.
- `safe_eval.py`: AST whitelist evaluator cho công thức số học.
- `aliases.py`: alias biomarker/biến công thức/unit.

Extractor bổ sung:

- `scripts/extract_thresholds_v2.py`: đọc `chunks.jsonl` và tạo `thresholds_extra.jsonl`.

## Chạy service

Từ repo root:

```bash
./scripts/run_medical_tools_service.sh
```

Biến môi trường optional:

```bash
MEDICAL_TOOLS_DATA_DIR=data/processed_data
MEDICAL_TOOLS_HOST=0.0.0.0
MEDICAL_TOOLS_PORT=8010
```

## Endpoints

### `GET /health`

Kiểm tra service sống.

### `GET /mcp/capabilities`

Trả danh sách biomarker threshold và formula mà tool hỗ trợ.

Alias nội bộ:

```text
GET /structured/capabilities
```

### `POST /mcp/medical-tools/evaluate`

Endpoint chính để MCP/AI service gọi.

Alias nội bộ:

```text
POST /structured/evaluate
```

Request ví dụ:

```json
{
  "text": "ACR 350 mg/g, GFR 55 ml/ph/1.73m2",
  "disease_name": "benh_than_man",
  "include_debug": false
}
```

Request với input đã parse sẵn:

```json
{
  "measurements": {
    "ACR": {"value": 350, "unit": "mg/g"},
    "GFR": {"value": 55, "unit": "ml/ph/1.73m2"}
  },
  "disease_name": "benh_than_man"
}
```

Request tính công thức:

```json
{
  "text": "nữ 60 tuổi, cân nặng 55 kg, chiều cao 160 cm, creatinine 1.4 mg/dL, race other"
}
```

Response chính gồm:

- `detected_measurements`: chỉ số lấy từ text/input.
- `derived_measurements`: chỉ số tính ra từ formula, ví dụ GFR từ MDRD.
- `threshold_matches`: tất cả ngưỡng khớp.
- `classifications`: ngưỡng có nhãn class/stage như `A3`, `G3a`, `CKD stage III`.
- `formula_results`: kết quả công thức hoặc danh sách input còn thiếu.
- `safety`: disclaimer và cảnh báo unit.

## Công thức đang hỗ trợ

Dựa trên `formulas.json` hiện tại:

- `ckd_epi_2021_creatinine`
- `mdrd_gfr`
- `cockcroft_gault`
- `body_surface_area`
- `fena_formula`

Lưu ý:
- `ckd_epi_2021_creatinine` là lựa chọn mặc định nên ưu tiên cho câu hỏi eGFR chung vì không cần biến `race`.
- `mdrd_gfr` vẫn được hỗ trợ khi người dùng gọi đích danh, và sẽ mặc định `race=other` nếu câu hỏi không cung cấp `race`; kết quả sẽ ghi rõ giả định đó trong `formula_results`.

Evaluator chỉ cho phép toán số học cơ bản qua AST whitelist, không cho phép function call, import, attribute access hoặc code execution.

## Luồng MCP đề xuất

```text
AI service / LangGraph
  -> phát hiện input có chỉ số xét nghiệm hoặc yêu cầu tính công thức
  -> MCP adapter gọi HTTP POST /mcp/medical-tools/evaluate
  -> nhận structured result
  -> AI service tổng hợp câu trả lời tự nhiên cho user
```

Service này chỉ trả structured result, không tự sinh câu trả lời LLM.

## Thay đổi gần đây

Phần này ghi lại các thay đổi mới để dễ hiểu vì sao chatbot đang hoạt động theo cách hiện tại.

### 1. Bỏ cơ chế `safe_structured_answer`

Trước đây graph của chatbot có một nhánh "an toàn":

- tool trả structured result
- graph dựng thêm một câu trả lời tóm tắt an toàn tên là `safe_structured_answer`
- trong một số tình huống, graph ưu tiên trả thẳng câu này thay vì để LLM tự diễn giải

Mục tiêu cũ là giảm hallucination, nhưng cơ chế này làm luồng hơi khó theo dõi và khiến việc debug cảm giác như tool "gọi được nhưng chưa chắc đã đi hết vào chatbot".

Hiện tại cơ chế đó đã được bỏ.

Luồng mới:

```text
tool result
  -> build_structured_context(...)
  -> đưa vào final prompt cùng evidence_context
  -> LLM tự viết câu trả lời cuối
```

Ý nghĩa:

- chatbot vẫn nhận dữ liệu từ medical tool service
- nhưng không còn bị ép phải trả một bản tóm tắt deterministic
- model có nhiều tự do hơn để diễn giải
- đổi lại, bạn chấp nhận rủi ro model có thể diễn giải sai hoặc nói rộng hơn facts

### 2. Tool vẫn là nơi tính toán chính

Việc bỏ `safe_structured_answer` không có nghĩa là bỏ tool.

Tool service vẫn chịu trách nhiệm:

- parse các chỉ số từ câu hỏi
- tính công thức như `CKD-EPI 2021`, `MDRD`, `Cockcroft-Gault`, `FENa`, `BSA`
- so ngưỡng và phân loại

Phần chatbot chỉ làm:

- gọi tool
- lấy kết quả đã chuẩn hóa
- đưa kết quả đó vào prompt
- để LLM tạo câu trả lời tự nhiên cho người dùng

Nói ngắn gọn:

- `service.py` quyết định số liệu
- `graph.py` quyết định đưa số liệu vào prompt thế nào
- LLM quyết định cách diễn đạt câu trả lời cuối

### 3. Ưu tiên CKD-EPI 2021 cho eGFR chung

Hiện tại:

- câu hỏi eGFR chung dùng `ckd_epi_2021_creatinine`
- công thức này không cần biến `race`
- `mdrd_gfr` chỉ còn là lựa chọn explicit khi user yêu cầu tính `MDRD`

Điều này tránh việc route mặc định phải dựa vào giả định `race=other`.

### 4. Sửa lỗi MDRD thiếu `race`

Một lỗi quan trọng trước đó là:

- câu hỏi eGFR kiểu `Nữ 60 tuổi, creatinine 1.4 mg/dL. Hãy ước tính eGFR.`
- tool nhận ra đây là câu hỏi công thức
- nhưng `mdrd_gfr` bị thiếu `race`, nên không tính ra số cuối cùng

Hiện tại:

- nếu `mdrd_gfr` thiếu `race`, service mặc định `race=other`
- kết quả vẫn được tính
- trong `formula_results` sẽ có ghi chú giả định này

Điều này giúp tool không còn dừng ở trạng thái `missing_inputs` cho các câu eGFR phổ biến.

### 5. Điều gì còn "an toàn" và điều gì không

Sau thay đổi này:

- vẫn còn sanitize `structured_context` để không nhét raw JSON nội bộ vào prompt
- vẫn còn `cleanup_user_answer(...)` để dọn bớt metadata nội bộ khỏi output cuối
- nhưng không còn lớp chặn nội dung kiểu "nếu model diễn giải quá tay thì ép quay về safe answer"

Vì vậy behavior hiện tại là:

- an toàn ở mức format và giấu metadata nội bộ
- không còn an toàn ở mức kiểm soát chặt nội dung diễn giải y khoa cuối

### 6. Khi nào nên nhớ điều này lúc debug

Nếu bạn thấy chatbot trả sai ở câu hỏi công thức/ngưỡng, hãy tách kiểm tra theo thứ tự:

1. Tool có được gọi không
2. Tool trả `formula_results` / `threshold_matches` đúng chưa
3. `structured_context` có chứa đúng dữ liệu không
4. Nếu 3 bước trên đúng mà answer cuối vẫn sai, lỗi nằm ở bước LLM diễn giải

Điểm mấu chốt:

- sai ở `service.py` là sai tính toán
- sai ở `structured_context` là sai khâu truyền dữ liệu vào prompt
- sai ở câu trả lời cuối dù `structured_context` đúng là sai do LLM diễn giải

## Thay đổi mới nhất về routing và payload tool

Để giảm lỗi khi gọi `medical_tool`, graph hiện có thêm một lớp tiền xử lý trước router:

```text
User query
  -> prepare_input
  -> extract_tool_payload
  -> route_with_medical_tools
  -> call_medical_tools
```

Ý nghĩa của `extract_tool_payload`:

- đọc câu hỏi và trích xuất sớm các field mà `medical_tool` thực sự hỗ trợ
- canonicalize tên field như `ACR`, `creatinine_mg_dl`, `weight_kg`, `urine_na`
- giữ lại `text` nguyên văn
- bỏ hẳn field không chắc hoặc không hỗ trợ
- không tạo `null`

Ví dụ trước đây có thể xuất hiện:

```json
{
  "text": "ACR 350 mg/g",
  "measurements": null,
  "disease_name": null,
  "formula_ids": [],
  "include_debug": false
}
```

Hiện tại payload mục tiêu là:

```json
{
  "text": "ACR 350 mg/g",
  "formula_ids": [],
  "include_debug": false
}
```

Nếu parse được chỉ số, payload sẽ có `measurements` thật:

```json
{
  "text": "Nữ 60 tuổi, creatinine 1.4 mg/dL, ACR 350 mg/g",
  "measurements": {
    "sex": {"value": "female"},
    "age": {"value": 60},
    "creatinine": {"value": 1.4, "unit": "mg/dL"},
    "creatinine_mg_dl": {"value": 1.4, "unit": "mg/dL"},
    "ACR": {"value": 350, "unit": "mg/g"}
  },
  "formula_ids": [],
  "include_debug": false
}
```

### Quy tắc payload hiện tại

- Không truyền `null` trong `tool_call.parameters`.
- `measurements` chỉ chứa field có trong medical tool API.
- Nếu graph đã extract được `measurements`, router phải ưu tiên dùng lại đúng payload đó.
- Nếu router/LLM trả thêm field lạ hoặc `formula_id` lạ, lớp sanitize sẽ loại bỏ trước khi gọi MCP.
- `formula_ids` luôn là list hợp lệ; khi không cần công thức thì dùng `[]`.

### Mục tiêu của thay đổi này

- giảm lỗi sai parameter khi gọi MCP server
- tránh việc router bịa field hoặc truyền `null` cho đủ schema
- tăng chất lượng dữ liệu đi vào `medical_tool`
- giúp `structured_context` cuối cùng đáng tin hơn cho bước answer synthesis

## Rebuild thresholds bổ sung

Khi `chunks.jsonl` thay đổi, chạy lại:

```bash
python3 scripts/extract_thresholds_v2.py \
  --input data/processed_data/chunks.jsonl \
  --base-thresholds data/processed_data/thresholds.jsonl \
  --output data/processed_data/thresholds_extra.jsonl
```

Hiện file extra bổ sung các nhóm rule thường bị extractor gốc bỏ sót:

- Hb/Hct
- pH/Bicarbonat
- HbA1c
- LDL cholesterol
- huyết áp mục tiêu
- phospho máu
- creatinine change μmol/L
- FENa
- Na/K nặng

## Lưu ý production

- Nên deploy service này như một container/API riêng.
- Không expose trực tiếp cho end-user nếu chưa có auth/rate limit.
- MCP adapter nên gọi endpoint này như một tool nội bộ.
- Nếu cần độ chính xác cao hơn, nên bổ sung unit conversion và validation schema riêng cho từng biomarker.
