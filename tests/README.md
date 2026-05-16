# Answer Test Suite

Thu muc nay chua bo test black-box cho chatbot QA cua VitalAI.

## Nhom test hien co

- `cases/general_qa.json`: hoi dap thong thuong
- `cases/threshold_qa.json`: hoi ve nguong/phan loai chi so
- `cases/formula_qa.json`: hoi can tinh cong thuc
- `cases/formula_threshold_qa.json`: hoi vua tinh cong thuc vua xem xet nguong

## Cach chay

Chay tat ca:

```bash
python3 tests/run_answer_tests.py
```

Chay 1 nhom:

```bash
python3 tests/run_answer_tests.py --category threshold_qa
```

Ghi report ra file khac:

```bash
python3 tests/run_answer_tests.py --output tests/results/my_report.md --json-output tests/results/my_report.json
```

Chay audit deterministic cho `medical_tools` (khong can LLM hay vector DB):

```bash
python3 tests/run_medical_tools_audit.py
```

Chay unit/API tests noi bo:

```bash
python3 -m unittest discover -s tests -v
```

`test_ai_service_api.py` kiem tra `POST /chat/answer` va SSE `POST /chat/stream` bang fake answerer, nen khong can goi LLM that khi backend dependencies da duoc cai.

## Yeu cau truoc khi chay

- `.env` da duoc cau hinh dung
- retrieval backend/index da san sang
- neu test lien quan formula/threshold qua tool service thi can chay:

```bash
./scripts/run_medical_tools_service.sh
```

## Output

Runner se ghi:

- `tests/results/answer_test_report.md`
- `tests/results/answer_test_report.json`

Markdown report de doc nhanh, JSON report de parse tu dong neu can.

Audit `medical_tools` se ghi them:

- `tests/results/medical_tools_audit_report.md`
- `tests/results/medical_tools_audit_report.json`

Bo audit nay tap trung vao cong thuc, boundary, parser, disease filter, unit conversion va classification de phat hien loi nghiep vu ma black-box answer test co the che mat.

## Ghi chu moi

Bo test formula/formula-threshold hien co them cac check debug cho luong tool:

- payload goi tool khong duoc chua `null`
- graph phai extract duoc mot so `measurements` hop le tu query truoc khi goi MCP

Muc tieu la khoa hanh vi moi: query co chi so -> extract payload sach -> sanitize router plan -> goi `medical_tool` voi parameter hop le.
