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
