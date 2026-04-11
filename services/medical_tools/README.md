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

- `mdrd_gfr`
- `cockcroft_gault`
- `body_surface_area`
- `fena_formula`

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
