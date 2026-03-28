# API Spec

## Mục tiêu của tài liệu này

File này mô tả API contract cho VitalAI ở giai đoạn đầu.

Nó trả lời:

1. version đầu cần những endpoint nào
2. request/response schema ra sao
3. response nào bắt buộc phải có citation
4. khi nào API trả limitation hoặc error

Đây là bước cầu nối giữa:

- `Agent_Graph_Spec.md`
- giai đoạn bắt đầu code backend

## Nguyên tắc cho API version đầu

Version đầu chỉ nên giải quyết:

- text input
- synchronous response
- một endpoint chat chính
- một endpoint health

Chưa nên làm ngay:

- streaming
- voice
- session persistence phức tạp
- upload file từ client

## API scope cho V1

### Bắt buộc có

1. `GET /health`
2. `POST /chat`

### Có thể thêm sau

1. `POST /chat/stream`
2. `POST /ingest`
3. `GET /sessions/{session_id}`

## Endpoint 1 — `GET /health`

### Mục tiêu

Kiểm tra service còn sống hay không.

### Response đề xuất

```json
{
  "status": "ok",
  "service": "vitalai",
  "version": "v1"
}
```

### Ghi chú

Endpoint này không cần đụng tới:

- retrieval
- agent graph
- database

ở version tối thiểu.

## Endpoint 2 — `POST /chat`

## Mục tiêu

Đây là endpoint trung tâm cho cả:

- medical question answering
- lab interpretation

## Request schema đề xuất

```json
{
  "message": "GFR bao nhiêu thì là bệnh thận mạn?",
  "session_id": "optional-session-id",
  "language": "vi",
  "include_debug": false
}
```

### Field bắt buộc

- `message`

### Field optional

- `session_id`
- `language`
- `include_debug`

### Rule cho `message`

- không được rỗng
- là text thuần
- không gửi PDF/file ở phase đầu

## Input classification tại API layer

API không cần tự xử lý toàn bộ logic phân loại.

Nhưng response nên luôn trả về:

- `input_type`

để client biết agent đã hiểu input là:

- `question`
- `lab_record`
- `mixed`

## Response schema đề xuất

```json
{
  "request_id": "uuid-or-deterministic-id",
  "session_id": "optional-session-id",
  "input_type": "question",
  "answer": "Bệnh thận mạn được định nghĩa khi GFR < 60 ml/ph/1.73m2 kéo dài trên 3 tháng.",
  "citations": [
    {
      "source_file": "ĐẠI CƯƠNG VỀ BỆNH LÝ CẦU THẬN( No Table) .pdf",
      "page": 133,
      "disease_name": "benh_than_man",
      "section_type": "definition"
    }
  ],
  "structured_findings": {
    "thresholds": [],
    "formulas": []
  },
  "safety": {
    "has_limitation": false,
    "flags": [],
    "disclaimer": "Thông tin chỉ mang tính tham khảo, không thay thế bác sĩ."
  },
  "debug": null
}
```

## Response bắt buộc phải có những gì

### Luôn có

- `input_type`
- `answer`
- `citations`
- `safety`

### Chỉ có khi phù hợp

- `structured_findings`
- `debug`

## Response cho `question`

### Ví dụ

```json
{
  "input_type": "question",
  "answer": "GFR < 60 ml/ph/1.73m2 kéo dài trên 3 tháng là tiêu chuẩn chẩn đoán bệnh thận mạn.",
  "citations": [
    {
      "source_file": "ĐẠI CƯƠNG VỀ BỆNH LÝ CẦU THẬN( No Table) .pdf",
      "page": 133
    }
  ],
  "structured_findings": {
    "thresholds": [
      {
        "biomarker": "GFR",
        "threshold_op": "<",
        "threshold_value": 60,
        "threshold_unit": "ml/ph/1.73m2"
      }
    ],
    "formulas": []
  },
  "safety": {
    "has_limitation": false,
    "flags": [],
    "disclaimer": "Thông tin chỉ mang tính tham khảo, không thay thế bác sĩ."
  }
}
```

## Response cho `lab_record`

### Ví dụ

```json
{
  "input_type": "lab_record",
  "answer": "ACR 350 mg/g phù hợp mức albumin niệu A3 theo phân loại hiện có trong tài liệu.",
  "citations": [
    {
      "source_file": "ĐẠI CƯƠNG VỀ BỆNH LÝ CẦU THẬN( No Table) .pdf",
      "page": 201
    }
  ],
  "structured_findings": {
    "thresholds": [
      {
        "biomarker": "ACR",
        "label": "A3",
        "threshold_op": ">",
        "threshold_value": 300,
        "threshold_unit": "mg/g"
      }
    ],
    "formulas": []
  },
  "safety": {
    "has_limitation": false,
    "flags": [],
    "disclaimer": "Thông tin chỉ mang tính tham khảo, không thay thế bác sĩ."
  }
}
```

## Field `citations`

Đây là field safety bắt buộc.

### Schema đề xuất

```json
{
  "source_file": "string",
  "page": 133,
  "disease_name": "benh_than_man",
  "section_type": "definition"
}
```

### Rule

- nếu có evidence thì phải có citation
- nếu không có citation thì phải có limitation rõ ràng

## Field `structured_findings`

Field này rất quan trọng cho lab interpretation.

### Gồm 2 nhóm

- `thresholds`
- `formulas`

### Mục tiêu

- cho client thấy API không chỉ trả text
- giúp frontend/rendering sau này dễ hơn
- giúp debug safety dễ hơn

## Field `safety`

Đây là field bắt buộc cho mọi response.

### Schema đề xuất

```json
{
  "has_limitation": true,
  "flags": [
    "insufficient_structured_evidence"
  ],
  "disclaimer": "Thông tin chỉ mang tính tham khảo, không thay thế bác sĩ."
}
```

### Các flag gợi ý

- `insufficient_structured_evidence`
- `weak_semantic_evidence`
- `out_of_scope_query`
- `limited_domain_coverage`
- `no_reliable_citation`

## Field `debug`

Chỉ nên trả khi:

- `include_debug = true`

### Mục đích

- hỗ trợ phát triển
- xem input_type
- xem query rewrite
- xem retrieval score

### Không nên đưa vào mặc định

Vì:

- tăng noise
- có thể làm lộ logic nội bộ không cần thiết

## Error handling

## 400 Bad Request

Khi:

- thiếu `message`
- `message` rỗng
- format JSON không hợp lệ

### Response đề xuất

```json
{
  "error": {
    "code": "bad_request",
    "message": "Field 'message' is required."
  }
}
```

## 422 Validation Error

Khi:

- request đúng JSON nhưng sai schema

## 500 Internal Server Error

Khi:

- agent failure
- retrieval failure chưa được handle
- backend exception

### Response đề xuất

```json
{
  "error": {
    "code": "internal_error",
    "message": "Unexpected server error."
  }
}
```

## Limitation response

Nếu query nằm ngoài độ phủ dữ liệu, API không nên fail bằng HTTP error.

Nó nên trả `200 OK` nhưng với:

- `answer` thận trọng
- `has_limitation = true`
- `flags` phù hợp

### Ví dụ

```json
{
  "input_type": "question",
  "answer": "Tài liệu hiện có chưa đủ dữ liệu đáng tin cậy để trả lời câu hỏi này ngoài phạm vi bệnh thận.",
  "citations": [],
  "structured_findings": {
    "thresholds": [],
    "formulas": []
  },
  "safety": {
    "has_limitation": true,
    "flags": [
      "out_of_scope_query",
      "limited_domain_coverage"
    ],
    "disclaimer": "Thông tin chỉ mang tính tham khảo, không thay thế bác sĩ."
  }
}
```

## Versioning

Version đầu chỉ cần implicit version trong docs.

Nếu mở rộng sau này, nên cân nhắc:

- `/api/v1/chat`

Nhưng không bắt buộc ngay ở giai đoạn đầu.

## Điều chưa làm ở phase này

Chưa làm:

- SSE streaming contract
- voice API
- upload API
- auth
- rate limit contract

## Kết luận

Sau file này, bước hợp lý tiếp theo có 2 hướng:

1. viết `Node_Prompt_Spec.md`
2. hoặc dừng phần docs và bắt đầu code từ ingestion

Nếu vẫn muốn khóa docs đầy đủ trước khi code, hướng hợp lý hơn là:

- viết `Node_Prompt_Spec.md`
