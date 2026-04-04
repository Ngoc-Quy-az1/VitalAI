from __future__ import annotations

"""
Bộ xử lí dữ liệu heuristic cho giai đoạn hiện tại.

Vì sao file này tồn tại:
- Repo hiện chỉ có một tài liệu nguồn là một file PDF thận học bị trộn nhiều loại nội dung.
- Trước khi làm retrieval hay agent, cần biến PDF này thành dữ liệu có cấu trúc để nhìn và kiểm tra được.
- File này là bước đầu tiên của quá trình đó.

File này tạo ra những gì:
- `chunks.jsonl`: các đoạn văn xuôi và các `text_chunks` hữu ích lấy lại từ JSON nhúng
- `thresholds.jsonl`: các ngưỡng số liệu lấy từ câu văn và từ các `rules` nhúng trong JSON
- `formulas.json`: các công thức được phục hồi từ phần JSON nhúng
- `summary.json`: thống kê nhanh để kiểm tra chất lượng extraction sau mỗi lần chạy

Giới hạn hiện tại:
- Đây vẫn là extractor heuristic, chưa phải ingestion pipeline cuối cùng.
- Mục tiêu hiện tại là dễ đọc, dễ debug và dễ kiểm tra thủ công.
- Việc thiếu `disease_name` ở nhiều đoạn overview là điều đã biết và sẽ được xử lí sau bằng
  heading/context propagation.
"""

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz


HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+.+$")
HEADING_PREFIX_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\.?\s+")
JSON_LINE_RE = re.compile(r'^\s*[{[]|^\s*"[^"]+"\s*:')
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
JSON_TEXT_RE = re.compile(r'"text"\s*:\s*"(?P<text>(?:\\.|[^"])*)"', re.DOTALL)
JSON_RULE_RE = re.compile(
    r'"if"\s*:\s*"(?P<if>(?:\\.|[^"])*)"\s*,\s*"then"\s*:\s*"(?P<then>(?:\\.|[^"])*)"',
    re.DOTALL,
)
JSON_ID_RE = re.compile(r'"id"\s*:\s*"(?P<value>(?:\\.|[^"])*)"', re.DOTALL)
JSON_NAME_RE = re.compile(r'"name"\s*:\s*"(?P<value>(?:\\.|[^"])*)"', re.DOTALL)
JSON_OUTPUT_RE = re.compile(r'"output"\s*:\s*"(?P<value>(?:\\.|[^"])*)"', re.DOTALL)
JSON_UNIT_RE = re.compile(r'"unit"\s*:\s*"(?P<value>(?:\\.|[^"])*)"', re.DOTALL)
JSON_INPUTS_RE = re.compile(r'"inputs"\s*:\s*\[(?P<value>.*?)\]', re.DOTALL)
JSON_INPUT_ITEM_RE = re.compile(r'"(?P<value>[^"]+)"')
FORMULA_RE = re.compile(r'"formula"\s*:\s*"(?P<value>(?:\\.|[^"])*)"', re.DOTALL)
INLINE_VALUE_RE = re.compile(
    r"(?P<op><=|>=|<|>|cao trên|tăng trên|giảm dưới|trên|dưới)\s*"
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>g/24 giờ|g/24h|g/l|g/L|mmol/l|mmol/L|mg/g|mg/mmol|mg/dL|ml/phút/1\.73m2|ml/ph/1\.73m2|ml/ph/1,73m2|ml/phút/1,73m2|ml/phút|ml/ph)",
    re.IGNORECASE,
)

SECTION_KEYWORDS = {
    "definition": ["khai niem", "dinh nghia"],
    "classification": ["phan loai", "kdigo", "giai doan", "stage"],
    "clinical_features": ["lam sang", "can lam sang", "trieu chung"],
    "diagnosis_criteria": ["chan doan", "tieu chuan chan doan"],
    "pathology": ["mo benh hoc", "ton thuong mo benh hoc", "kinh hien vi", "mien dich huynh quang", "hien vi dien tu"],
    "treatment": ["dieu tri", "nguyen tac dieu tri", "lieu dung", "thuoc"],
    "progression": ["tien trien", "tien luong"],
    "complications": ["bien chung"],
    "follow_up": ["theo doi", "tai kham", "phong ngua", "du phong"],
}

DISEASE_KEYWORDS = {
    "benh_ly_cau_than": [
        "benh ly cau than",
        "benh cau than nguyen phat",
        "benh cau than thu phat",
        "phan loai benh cau than",
        "benh cau than giai doan cuoi",
        "ton thuong cau than sau ghep",
        "chan doan mot so benh cau than thuong gap",
    ],
    "benh_than_man": [
        "benh than man",
        "suy than man",
        "benh cau than man",
        "chronic kidney disease",
        "ckd",
        "kdigo",
    ],
    "lupus_nephritis": ["lupus", "viem cau than lupus"],
    "acute_kidney_injury": [
        "suy than cap",
        "ton thuong than cap",
        "acute kidney injury",
        "aki",
        "suy than cap truoc than",
        "suy than cap sau than",
        "suy than cap tai than",
    ],
    "hoi_chung_than_hu": ["hoi chung than hu"],
    "viem_cau_than_cap": ["viem cau than cap"],
    "viem_cau_than_man": ["viem cau than man"],
    "viem_cau_than_tien_trien_nhanh": ["viem cau than tien trien nhanh", "tien trien nhanh"],
    "benh_than_iga": ["benh than iga", "iga nephropathy", "iga"],
    "diabetic_kidney_disease": [
        "dai thao duong",
        "diabetic kidney disease",
        "than dai thao duong",
        "benh than dtd",
        "benh than do dtd",
        "dtd",
    ],
}

BIOMARKER_ALIASES = {
    "protein_niệu_24h": ["protein niu", "protein nieu", "dam niu", "dam nieu"],
    "protein_mau": ["protein mau", "protein huyet"],
    "albumin_máu": ["albumin mau", "albumin huyet"],
    "cholesterol": ["cholesterol"],
    "ACR": [
        "ty le albumin/creatinine",
        "ty le albumin/creatinin",
        "albumin/creatinine",
        "albumin/creatinin",
        "albumine/creatinine",
        "albumine/creatinin",
        "acr",
        "albumin creatinine ratio",
        "albumin-creatinine",
    ],
    "PCR": [
        "ty le protein/creatinine",
        "ty le protein/creatinin",
        "ty so protein/creatinine",
        "ty so protein/creatinin",
        "ty le protein nieu/creatinine",
        "ty le protein nieu/creatinin",
        "ty so protein nieu/creatinine",
        "ty so protein nieu/creatinin",
        "ty le protein nieu/creatinin nieu",
        "ty so protein nieu/creatinin nieu",
        "protein/creatinine",
        "protein/creatinin",
        "protein niu/creatinin niu",
        "protein nieu/creatinin niu",
        "protein nieu/creatinin nieu",
        "protein niu/creatinine",
        "protein nieu/creatinine",
    ],
    "creatinine": ["creatinin", "creatinine"],
    "GFR": ["gfr", "egfr", "muc loc cau than", "mlct"],
    "BSA": ["dien tich da", "bsa"],
    "FENa": ["fena", "fractional excretion of sodium"],
    "sodium": ["na+", "natri"],
    "urea": ["ure mau", "urea", "ure"],
}

THRESHOLD_UNIT_DEFAULTS = {
    "protein_niệu_24h": "g/24h",
    "protein_mau": "g/L",
    "albumin_máu": "g/L",
    "cholesterol": "mmol/L",
    "PCR": "mg/g",
    "creatinine": "mg/dL",
    "GFR": "ml/ph/1.73m2",
    "ACR": "mg/g",
    "sodium": "mmol/L",
    "urea": "mmol/L",
}

DISEASE_PRIORITY = {
    "benh_ly_cau_than": 10,
    "benh_than_man": 20,
    "acute_kidney_injury": 30,
    "hoi_chung_than_hu": 80,
    "viem_cau_than_cap": 80,
    "viem_cau_than_man": 80,
    "viem_cau_than_tien_trien_nhanh": 80,
    "benh_than_iga": 80,
    "lupus_nephritis": 90,
    "diabetic_kidney_disease": 90,
}

CONTEXT_RESET_MARKERS = (
    "dai cuong ve benh ly cau than",
    "phan loai dua vao lam sang",
    "phan loai benh cau than dua vao ton thuong",
    "benh cau than nguyen phat",
    "benh cau than thu phat",
    "cac benh ly cau than khac",
    "benh cau than giai doan cuoi",
    "ton thuong cau than sau ghep",
    "chan doan mot so benh cau than thuong gap",
)

HEADING_PRESERVING_SPLIT_THRESHOLD = 1100
SUBCHUNK_TARGET_LENGTH = 900
SUBCHUNK_MAX_UNITS = 2


@dataclass
class PageRecord:
    """Cấu trúc tối giản để giữ nội dung text của từng trang PDF."""

    page: int
    text: str


class MedicalPdfProcessor:
    """
    Bộ điều phối chính cho việc chuyển PDF sang dữ liệu có cấu trúc.

    Luồng xử lí được giữ đơn giản có chủ đích:
    1. Đọc text thô từ từng trang PDF.
    2. Tách phần văn xuôi thành các chunk.
    3. Trích các ngưỡng số liệu từ câu văn và từ `rules` trong JSON nhúng.
    4. Khôi phục các công thức tường minh từ các block JSON nhúng.
    5. Ghi toàn bộ ra file để có thể đọc và kiểm tra thủ công.
    """

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)
        self.source_file = self.pdf_path.name

    def process(self) -> dict[str, Any]:
        """Chạy toàn bộ pipeline extraction trong bộ nhớ và trả về tất cả artifact."""

        pages = self._extract_pages()
        extracted_chunks = self._extract_chunks(pages)
        propagated_chunks = self._propagate_missing_chunk_disease_names(extracted_chunks)
        chunks = self._dedupe_by_key(propagated_chunks, "chunk_id")
        thresholds = self._dedupe_by_key(self._extract_thresholds(pages, chunks), "threshold_id")
        formulas = self._dedupe_by_key(self._extract_formulas(pages), "formula_id")

        return {
            "chunks": chunks,
            "thresholds": thresholds,
            "formulas": formulas,
            "summary": self._build_summary(pages, chunks, thresholds, formulas),
        }

    def _extract_pages(self) -> list[PageRecord]:
        """Đọc từng trang PDF thành plain text. Bước này cố tình giữ đơn giản."""

        document = fitz.open(self.pdf_path)
        return [
            PageRecord(page=index + 1, text=document.load_page(index).get_text("text"))
            for index in range(document.page_count)
        ]

    def _extract_chunks(self, pages: list[PageRecord]) -> list[dict[str, Any]]:
        """
        Tạo các chunk dùng cho retrieval về sau.

        Hiện giữ lại 2 nguồn chunk:
        - các block văn xuôi nhìn thấy trực tiếp trong PDF
        - các `text_chunks` lấy lại từ JSON nhúng trong PDF
        """

        chunks: list[dict[str, Any]] = []
        current_disease_context: str | None = None

        for page_record in pages:
            prose_text = self._remove_json_lines(page_record.text)
            page_chunks = self._split_prose_blocks(prose_text)
            for chunk_index, chunk_text in enumerate(page_chunks, start=1):
                explicit_disease = self._detect_disease_name(chunk_text)
                if self._is_context_reset_text(chunk_text):
                    current_disease_context = None

                if self._is_disease_anchor_text(chunk_text, explicit_disease):
                    current_disease_context = explicit_disease

                resolved_disease = self._resolve_disease_name(
                    text=chunk_text,
                    explicit_disease=explicit_disease,
                    current_disease_context=current_disease_context,
                )
                metadata = self._build_metadata(
                    text=chunk_text,
                    page=page_record.page,
                    chunk_index=chunk_index,
                    content_type="prose",
                    disease_name=resolved_disease,
                )
                chunks.append(
                    {
                        "chunk_id": self._make_chunk_id(metadata["disease_name"], page_record.page, chunk_index),
                        "content": chunk_text,
                        "metadata": metadata,
                    }
                )

            json_chunk_index = len(page_chunks) + 1
            for extracted_text in self._extract_json_text_chunks(page_record.text):
                cleaned = self._clean_json_string(extracted_text)
                if len(cleaned) < 20:
                    continue
                explicit_disease = self._detect_disease_name(cleaned)
                resolved_disease = explicit_disease or current_disease_context
                metadata = self._build_metadata(
                    text=cleaned,
                    page=page_record.page,
                    chunk_index=json_chunk_index,
                    content_type="json_block",
                    disease_name=resolved_disease,
                )
                chunks.append(
                    {
                        "chunk_id": self._make_chunk_id(metadata["disease_name"], page_record.page, json_chunk_index),
                        "content": cleaned,
                        "metadata": metadata,
                    }
                )
                json_chunk_index += 1

        return chunks

    def _propagate_missing_chunk_disease_names(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Điền `disease_name` cho các chunk còn thiếu bằng context heading lân cận.

        Đây là pass hậu xử lí cho đúng cấu trúc của tài liệu hiện có:
        nhiều mục con như `5.1`, `7.1`, `4.4.8.1` không nhắc lại tên bệnh,
        nhưng lại nằm ngay dưới một heading cha hoặc một chunk anh em đã biết disease.
        """

        page_context: dict[int, str] = {}
        for item in chunks:
            disease_name = item["metadata"].get("disease_name")
            if disease_name is not None:
                page_context.setdefault(item["metadata"]["page"], disease_name)

        for index, item in enumerate(chunks):
            metadata = item["metadata"]
            if metadata.get("disease_name") is not None:
                continue

            heading_number = self._extract_heading_number(item["content"])
            propagated = self._infer_disease_from_neighbor_chunks(chunks, index, heading_number)

            if propagated is None and heading_number is None:
                propagated = page_context.get(metadata["page"]) or page_context.get(metadata["page"] - 1)

            if propagated is None:
                continue

            metadata["disease_name"] = propagated
            chunk_id = item["chunk_id"]
            if chunk_id.startswith("unknown_"):
                item["chunk_id"] = self._make_chunk_id(propagated, metadata["page"], metadata["chunk_index"])
            page_context.setdefault(metadata["page"], propagated)

        return chunks

    def _infer_disease_from_neighbor_chunks(
        self,
        chunks: list[dict[str, Any]],
        current_index: int,
        heading_number: str | None,
    ) -> str | None:
        """
        Suy luận disease từ các chunk gần kề có cùng nhánh heading.

        Ví dụ:
        - `7.1` có thể kế thừa từ `7.2.1`
        - `5.2` có thể kế thừa từ chapter `5`
        - `4.4.8.1` có thể kế thừa từ `4.4.8`
        """

        current_page = chunks[current_index]["metadata"]["page"]
        candidate_prefixes = self._candidate_heading_prefixes(heading_number)
        best_match: tuple[int, int, str] | None = None
        nearby_diseases: list[str] = []

        for offset in range(1, 16):
            for neighbor_index in (current_index - offset, current_index + offset):
                if neighbor_index < 0 or neighbor_index >= len(chunks):
                    continue

                neighbor = chunks[neighbor_index]
                neighbor_disease = neighbor["metadata"].get("disease_name")
                if neighbor_disease is None:
                    continue

                neighbor_page = neighbor["metadata"]["page"]
                if abs(neighbor_page - current_page) > 1:
                    continue

                neighbor_heading = self._extract_heading_number(neighbor["content"])
                if heading_number is not None and not candidate_prefixes and offset <= 6 and neighbor_heading is not None:
                    nearby_diseases.append(neighbor_disease)

                if heading_number is None:
                    if neighbor_page == current_page or neighbor_index < current_index:
                        return neighbor_disease
                    continue

                match_length = self._shared_heading_match_length(candidate_prefixes, neighbor_heading)
                if match_length == 0:
                    continue

                if best_match is None or match_length > best_match[0] or (
                    match_length == best_match[0] and offset < best_match[1]
                ):
                    best_match = (match_length, offset, neighbor_disease)

        if heading_number is not None and not candidate_prefixes:
            unique_nearby = list(dict.fromkeys(nearby_diseases))
            if len(unique_nearby) == 1:
                return unique_nearby[0]

        return best_match[2] if best_match else None

    def _candidate_heading_prefixes(self, heading_number: str | None) -> list[str]:
        """Sinh các prefix heading cha để phục vụ propagate context."""

        if heading_number is None:
            return []
        parts = heading_number.split(".")
        return [".".join(parts[:length]) for length in range(len(parts) - 1, 0, -1)]

    def _shared_heading_match_length(self, candidate_prefixes: list[str], neighbor_heading: str | None) -> int:
        """Đo độ khớp giữa heading hiện tại và heading của chunk lân cận."""

        if neighbor_heading is None:
            return 0
        for prefix in candidate_prefixes:
            if neighbor_heading == prefix or neighbor_heading.startswith(f"{prefix}."):
                return len(prefix.split("."))
        return 0

    def _extract_thresholds(
        self,
        pages: list[PageRecord],
        chunks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Trích xuất các ngưỡng số liệu có cấu trúc.

        Nguồn lấy dữ liệu gồm:
        - câu văn bình thường như `protein niệu > 3,5 g/24h`
        - các block `rules` nhúng trong JSON như rule phân tầng KDIGO / ACR
        """

        thresholds: list[dict[str, Any]] = []
        page_disease_context = self._build_page_disease_context(chunks or [])
        current_disease_context: str | None = None

        for page_record in pages:
            prose_text = self._remove_json_lines(page_record.text)
            page_explicit_disease = self._detect_disease_name(prose_text)
            if self._is_context_reset_text(prose_text):
                current_disease_context = None
            if page_explicit_disease and self._is_disease_anchor_text(prose_text, page_explicit_disease):
                current_disease_context = page_explicit_disease

            disease_name = page_explicit_disease or page_disease_context.get(page_record.page) or current_disease_context
            section_type = self._detect_section_type(prose_text)

            for index, sentence in enumerate(self._split_sentences(prose_text), start=1):
                explicit_sentence_disease = self._detect_disease_name(sentence)
                if self._is_disease_anchor_text(sentence, explicit_sentence_disease):
                    current_disease_context = explicit_sentence_disease

                thresholds.extend(
                    self._extract_thresholds_from_sentence(
                        sentence=sentence,
                        page=page_record.page,
                        disease_name=explicit_sentence_disease or disease_name or current_disease_context,
                        section_type=section_type,
                        base_index=index,
                    )
                )

            for index, rule in enumerate(self._extract_json_rules(page_record.text), start=1):
                biomarker = self._extract_biomarker_from_rule(rule["if"]) or self._detect_biomarker(page_record.text)
                if biomarker is None:
                    continue

                min_value, max_value, op = self._parse_rule_condition(rule["if"])
                if min_value is None and max_value is None:
                    continue

                threshold_value = min_value if min_value is not None else max_value
                resolved_rule_disease = (
                    self._detect_disease_name(page_record.text)
                    or page_disease_context.get(page_record.page)
                    or page_disease_context.get(page_record.page - 1)
                    or page_disease_context.get(page_record.page + 1)
                    or current_disease_context
                )
                threshold: dict[str, Any] = {
                    "threshold_id": self._make_threshold_id(
                        disease_name=resolved_rule_disease,
                        biomarker=biomarker,
                        operator=op,
                        value=str(threshold_value),
                        page=page_record.page,
                        suffix=index,
                    ),
                    "biomarker": biomarker,
                    "threshold_op": op,
                    "threshold_value": threshold_value,
                    "threshold_unit": self._infer_threshold_unit(page_record.text, biomarker),
                    "label": rule["then"],
                    "severity": None,
                    "disease_name": resolved_rule_disease,
                    "section_type": "classification",
                    "content_type": "threshold_value",
                    "source_text": f'if {rule["if"]} then {rule["then"]}',
                    "source_file": self.source_file,
                    "page": page_record.page,
                    "language": "vi",
                }
                if op == "between":
                    threshold["threshold_value_min"] = min_value
                    threshold["threshold_value_max"] = max_value
                thresholds.append(threshold)

        return thresholds

    def _extract_formulas(self, pages: list[PageRecord]) -> list[dict[str, Any]]:
        """
        Khôi phục các công thức tường minh từ text mang hình dạng JSON.

        Quanh mỗi key `"formula"`, hàm sẽ cố lấy lại các field lân cận như `id`, `name`,
        `inputs`, `output`, `unit`. Việc match theo cửa sổ ngữ cảnh là cần thiết vì text PDF
        thường làm vỡ một object JSON qua nhiều dòng hoặc nhiều trang.
        """

        formulas: list[dict[str, Any]] = []

        for index, page_record in enumerate(pages):
            prev_text = pages[index - 1].text if index > 0 else ""
            next_text = pages[index + 1].text if index + 1 < len(pages) else ""
            context = f"{prev_text[-2000:]}\n{page_record.text}\n{next_text[:2000]}"
            current_offset = len(prev_text[-2000:]) + 1

            for formula_match in FORMULA_RE.finditer(page_record.text):
                relative_index = current_offset + formula_match.start()
                window_start = max(0, relative_index - 1600)
                window_end = min(len(context), relative_index + 1600)
                window = context[window_start:window_end]
                formula_index = relative_index - window_start
                before = window[:formula_index]
                after = window[formula_index:]

                formula_expr = self._clean_json_string(formula_match.group("value"))
                formula_id = self._extract_last_value(before, JSON_ID_RE)
                formula_name = self._extract_last_value(before, JSON_NAME_RE)
                output_name = self._extract_first_value(after, JSON_OUTPUT_RE)
                output_unit = self._extract_first_value(after, JSON_UNIT_RE)
                inputs = self._extract_inputs(before) or self._extract_inputs(window)

                if not formula_name and not formula_id:
                    continue
                if not self._looks_like_formula_expression(formula_expr):
                    continue

                formula_text = self._extract_formula_source_text(window, formula_index, formula_expr)
                has_stage_name = bool(formula_name and "stage" in formula_name.lower())
                formula_type = "classification" if has_stage_name else "calculation"

                formulas.append(
                    {
                        "formula_id": self._slugify(formula_id or formula_name or formula_expr),
                        "formula_name": formula_name or formula_id,
                        "formula_type": formula_type,
                        "expression": formula_expr,
                        "variables": [
                            {
                                "name": item,
                                "description": item.replace("_", " "),
                                "unit": None,
                                "required": True,
                            }
                            for item in inputs
                        ],
                        "output_name": output_name or formula_name or formula_id,
                        "output_unit": self._normalize_unit(output_unit) if output_unit else None,
                        "disease_name": self._infer_formula_disease_name(
                            formula_name=formula_name or formula_id,
                            text=window,
                        ),
                        "section_type": self._detect_section_type(window),
                        "source_text": formula_text,
                        "source_file": self.source_file,
                        "page": page_record.page,
                        "language": "vi",
                    }
                )

        return formulas

    def _split_prose_blocks(self, text: str) -> list[str]:
        """Tách block văn xuôi theo heading; đây mới là giải pháp tạm trước semantic chunking."""

        blocks: list[list[str]] = []
        current: list[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if current:
                    current.append("")
                continue

            if HEADING_RE.match(line) and current:
                blocks.append(current)
                current = [line]
            else:
                current.append(line)

        if current:
            blocks.append(current)

        cleaned: list[str] = []
        for lines in blocks:
            block = "\n".join(lines).strip()
            if len(block) < 30:
                continue
            cleaned.extend(self._split_large_prose_block(block))
        return cleaned

    def _split_large_prose_block(self, block: str) -> list[str]:
        """
        Tách prose block quá dài theo bullet/đoạn nhưng vẫn giữ heading ở đầu chunk.

        Mục tiêu là tránh một chunk overview quá dài gom nhiều ý khác nhau,
        làm embedding bị loãng và đè mất ý định nghĩa/khái niệm.
        """

        if len(block) < HEADING_PRESERVING_SPLIT_THRESHOLD:
            return [block]

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            return [block]

        heading = lines[0] if HEADING_RE.match(lines[0]) else None
        body_lines = lines[1:] if heading else lines
        units = self._split_block_units(body_lines)
        if len(units) < 3:
            return [block]

        chunks: list[str] = []
        current_units: list[str] = []
        current_length = len(heading or "")

        for unit in units:
            projected_length = current_length + len(unit)
            if current_units and (
                len(current_units) >= SUBCHUNK_MAX_UNITS or projected_length > SUBCHUNK_TARGET_LENGTH
            ):
                chunks.append(self._compose_prose_subchunk(heading, current_units))
                current_units = [unit]
                current_length = len(heading or "") + len(unit)
                continue

            current_units.append(unit)
            current_length = projected_length

        if current_units:
            chunks.append(self._compose_prose_subchunk(heading, current_units))

        return chunks if len(chunks) > 1 else [block]

    def _split_block_units(self, lines: list[str]) -> list[str]:
        """Nhóm các dòng thuộc cùng một bullet/đoạn để phục vụ chia subchunk."""

        units: list[str] = []
        current: list[str] = []

        for line in lines:
            if self._is_bullet_like_line(line):
                if current:
                    units.append("\n".join(current).strip())
                current = [line]
                continue

            if not current:
                current = [line]
                continue

            current.append(line)

        if current:
            units.append("\n".join(current).strip())

        return [unit for unit in units if unit]

    def _compose_prose_subchunk(self, heading: str | None, units: list[str]) -> str:
        """Ghép heading với các đơn vị nội dung con thành một chunk hoàn chỉnh."""

        if heading is None:
            return "\n".join(units).strip()
        return "\n".join([heading] + units).strip()

    def _is_bullet_like_line(self, line: str) -> bool:
        stripped = line.lstrip()
        return stripped.startswith(("–", "-", "•"))

    def _split_sentences(self, text: str) -> list[str]:
        """Tách câu phục vụ extract threshold. Giữ bảo thủ để tránh cắt quá vụn."""

        sentences = []
        for item in SENTENCE_SPLIT_RE.split(text):
            sentence = " ".join(item.replace("\n", " ").split()).strip()
            if len(sentence) >= 20:
                sentences.append(sentence)
        return sentences

    def _remove_json_lines(self, text: str) -> str:
        """Loại các dòng nhìn rõ là JSON trước khi chunk phần văn xuôi."""

        cleaned_lines = []
        for line in text.splitlines():
            if JSON_LINE_RE.match(line.strip()):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _extract_json_text_chunks(self, text: str) -> list[str]:
        return [match.group("text") for match in JSON_TEXT_RE.finditer(text)]

    def _extract_json_rules(self, text: str) -> list[dict[str, str]]:
        return [
            {
                "if": self._clean_json_string(match.group("if")),
                "then": self._clean_json_string(match.group("then")),
            }
            for match in JSON_RULE_RE.finditer(text)
        ]

    def _extract_thresholds_from_sentence(
        self,
        sentence: str,
        page: int,
        disease_name: str | None,
        section_type: str,
        base_index: int,
    ) -> list[dict[str, Any]]:
        """
        Trích một hoặc nhiều threshold từ một câu.

        Việc gán biomarker được làm theo thứ tự ưu tiên:
        - nhìn phần text bên trái ngưỡng số
        - nếu không đủ, suy từ đơn vị
        - chỉ fallback sang cả câu khi trong câu chỉ có đúng một biomarker rõ ràng
        """

        items: list[dict[str, Any]] = []
        local_disease_name = self._detect_disease_name(sentence) or disease_name
        detected_section = self._detect_section_type(sentence)
        local_section_type = detected_section if detected_section != "general" else section_type

        for local_index, match in enumerate(INLINE_VALUE_RE.finditer(sentence), start=1):
            biomarker = self._infer_threshold_biomarker(sentence, match.start(), match.group("unit"))
            if biomarker is None:
                continue

            item = {
                "threshold_id": self._make_threshold_id(
                    disease_name=local_disease_name,
                    biomarker=biomarker,
                    operator=self._normalize_operator(match.group("op")),
                    value=match.group("value"),
                    page=page,
                    suffix=(base_index * 100) + local_index,
                ),
                "biomarker": biomarker,
                "threshold_op": self._normalize_operator(match.group("op")),
                "threshold_value": self._to_float(match.group("value")),
                "threshold_unit": self._normalize_unit(match.group("unit")),
                "label": None,
                "severity": None,
                "disease_name": local_disease_name,
                "section_type": local_section_type,
                "content_type": "threshold_value",
                "source_text": sentence,
                "source_file": self.source_file,
                "page": page,
                "language": "vi",
            }
            items.append(item)

        return items

    def _extract_last_value(self, text: str, pattern: re.Pattern[str]) -> str | None:
        matches = list(pattern.finditer(text))
        if not matches:
            return None
        return self._clean_json_string(matches[-1].group("value"))

    def _extract_first_value(self, text: str, pattern: re.Pattern[str]) -> str | None:
        match = pattern.search(text)
        if not match:
            return None
        return self._clean_json_string(match.group("value"))

    def _extract_inputs(self, text: str) -> list[str]:
        matches = list(JSON_INPUTS_RE.finditer(text))
        if not matches:
            return []
        raw = matches[-1].group("value")
        return [self._clean_json_string(item.group("value")) for item in JSON_INPUT_ITEM_RE.finditer(raw)]

    def _extract_formula_source_text(self, window: str, formula_index: int, formula_expr: str) -> str:
        """Giữ lại một đoạn context ngắn để sau này audit công thức lấy từ đâu ra."""

        left = max(0, formula_index - 200)
        right = min(len(window), formula_index + len(formula_expr) + 200)
        return " ".join(window[left:right].split())

    def _looks_like_formula_expression(self, value: str) -> bool:
        return any(token in value for token in ("+", "-", "*", "/", "(", ")", "**"))

    def _build_metadata(
        self,
        text: str,
        page: int,
        chunk_index: int,
        content_type: str,
        disease_name: str | None = None,
    ) -> dict[str, Any]:
        """Lắp metadata theo contract đã chốt cho downstream retrieval về sau."""

        section_type = self._detect_section_type(text)
        biomarker = self._detect_biomarker(text)
        return {
            "doc_type": self._detect_doc_type(
                text,
                content_type,
                section_type=section_type,
                biomarker=biomarker,
            ),
            "disease_name": disease_name if disease_name is not None else self._detect_disease_name(text),
            "section_type": section_type,
            "content_type": content_type,
            "biomarker": biomarker,
            "source_file": self.source_file,
            "page": page,
            "language": "vi",
            "chunk_index": chunk_index,
        }

    def _detect_doc_type(
        self,
        text: str,
        content_type: str,
        section_type: str | None = None,
        biomarker: str | None = None,
    ) -> str:
        normalized = self._normalize_ascii(text)
        resolved_section = section_type or self._detect_section_type(text)
        if content_type == "json_block" and any(key in normalized for key in ("function_mapping", "formula", "inputs")):
            return "formula_reference"
        if any(key in normalized for key in ("formula", "cockcroft", "mdrd", "fena")):
            return "formula_reference"
        if biomarker and resolved_section in {"classification", "diagnosis_criteria"} and any(
            key in normalized for key in ("gfr", "acr", "pcr", "kdigo", "microalbuminuria", "protein niu", "albumin")
        ):
            return "threshold_reference"
        if resolved_section == "treatment" and any(
            key in normalized for key in ("lieu dung", "cyclophosphamid", "prednisone", "methylprednisolone")
        ):
            return "medication_reference"
        return "disease_guideline"

    def _detect_disease_name(self, text: str) -> str | None:
        normalized = self._normalize_ascii(text)
        matches = [
            disease_name
            for disease_name, keywords in DISEASE_KEYWORDS.items()
            if any(self._contains_keyword(normalized, keyword) for keyword in keywords)
        ]
        unique_matches = list(dict.fromkeys(matches))
        if len(unique_matches) == 1:
            return unique_matches[0]
        if not unique_matches:
            return None
        if "benh_ly_cau_than" in unique_matches and len(unique_matches) >= 3:
            return "benh_ly_cau_than"

        ranked_matches = sorted(
            unique_matches,
            key=lambda disease_name: DISEASE_PRIORITY.get(disease_name, 50),
            reverse=True,
        )
        top_match = ranked_matches[0]
        top_priority = DISEASE_PRIORITY.get(top_match, 50)
        same_priority = [
            disease_name
            for disease_name in ranked_matches
            if DISEASE_PRIORITY.get(disease_name, 50) == top_priority
        ]
        if len(same_priority) == 1:
            return top_match
        return None

    def _detect_section_type(self, text: str) -> str:
        normalized = self._normalize_ascii(text)
        lines = [self._normalize_ascii(line.strip()) for line in text.splitlines() if line.strip()]
        heading_preview = " ".join(lines[:2])
        section_preview = " ".join(lines[:6]) if lines else normalized[:400]

        for section_type in (
            "definition",
            "classification",
            "pathology",
            "treatment",
            "diagnosis_criteria",
            "clinical_features",
            "progression",
            "complications",
            "follow_up",
        ):
            if any(keyword in heading_preview for keyword in SECTION_KEYWORDS[section_type]):
                return section_type

        for section_type in (
            "definition",
            "classification",
            "pathology",
            "treatment",
            "diagnosis_criteria",
            "clinical_features",
            "progression",
            "complications",
            "follow_up",
        ):
            if any(keyword in section_preview for keyword in SECTION_KEYWORDS[section_type]):
                return section_type
        return "general"

    def _detect_biomarker(self, text: str) -> str | None:
        normalized = self._normalize_ascii(text)
        return self._find_last_biomarker_before(normalized)

    def _infer_threshold_biomarker(self, sentence: str, match_start: int, raw_unit: str) -> str | None:
        """
        Suy luận threshold này thuộc biomarker nào.

        Hàm này cần thiết vì câu y khoa thường chứa nhiều biomarker cùng lúc,
        ví dụ `protein máu < 60` và `albumin máu < 30` trong cùng một câu.
        """

        context_before = self._normalize_ascii(sentence[:match_start])
        ratio_biomarker = self._detect_ratio_biomarker(context_before)
        if ratio_biomarker:
            return ratio_biomarker

        before_match = self._find_last_biomarker_before(context_before)
        if before_match:
            return before_match

        unit = self._normalize_unit(raw_unit)
        unit_hint_map = {
            "g/24h": "protein_niệu_24h",
            "ml/ph/1.73m2": "GFR",
            "ml/ph": "GFR",
        }
        hinted = unit_hint_map.get(unit)
        if hinted:
            return hinted

        normalized_sentence = self._normalize_ascii(sentence)
        ratio_biomarker = self._detect_ratio_biomarker(normalized_sentence)
        if ratio_biomarker:
            return ratio_biomarker

        if unit in {"mg/g", "mg/mmol"}:
            if "albumin" in normalized_sentence or "albumine" in normalized_sentence:
                return "ACR"
            if "protein" in normalized_sentence:
                return "PCR"

        candidates = [
            biomarker
            for biomarker, aliases in BIOMARKER_ALIASES.items()
            if any(alias in normalized_sentence for alias in aliases)
        ]
        unique_candidates = list(dict.fromkeys(candidates))
        if len(unique_candidates) == 1:
            return unique_candidates[0]
        return None

    def _find_last_biomarker_before(self, normalized_text: str) -> str | None:
        ratio_biomarker = self._detect_ratio_biomarker(normalized_text)
        if ratio_biomarker:
            return ratio_biomarker

        best_match: tuple[int, int, str] | None = None
        for biomarker, aliases in BIOMARKER_ALIASES.items():
            for alias in aliases:
                index = self._rfind_keyword(normalized_text, alias)
                if index == -1:
                    continue
                end_index = index + len(alias)
                alias_length = len(alias)
                if best_match is None or end_index > best_match[0] or (
                    end_index == best_match[0] and alias_length > best_match[1]
                ):
                    best_match = (end_index, alias_length, biomarker)
        return best_match[2] if best_match else None

    def _extract_biomarker_from_rule(self, condition: str) -> str | None:
        normalized = self._normalize_ascii(condition)
        ratio_biomarker = self._detect_ratio_biomarker(normalized)
        if ratio_biomarker:
            return ratio_biomarker
        return self._find_last_biomarker_before(normalized)

    def _detect_ratio_biomarker(self, normalized_text: str) -> str | None:
        """
        Ưu tiên nhận diện biomarker kiểu tỷ lệ trước khi fallback về biomarker đơn lẻ.

        Mục đích là tránh trường hợp `protein niệu/creatinin niệu`
        bị gán nhầm thành `creatinine` chỉ vì `creatinin` đứng cuối cụm.
        """

        ratio_patterns = {
            "ACR": [
                r"(ty le|ty so)\s+albumin(?:e)?(?:\s+nieu)?\s*/\s*creatin(?:in|ine)(?:\s+nieu)?",
                r"albumin(?:e)?(?:\s+nieu)?\s*/\s*creatin(?:in|ine)(?:\s+nieu)?",
                r"albumin[-\s]*creatinine",
            ],
            "PCR": [
                r"(ty le|ty so)\s+protein(?:\s+nieu)?\s*/\s*creatin(?:in|ine)(?:\s+nieu)?",
                r"protein(?:\s+nieu)?\s*/\s*creatin(?:in|ine)(?:\s+nieu)?",
                r"protein[-\s]*creatinine",
            ],
        }
        for biomarker, patterns in ratio_patterns.items():
            if any(re.search(pattern, normalized_text) for pattern in patterns):
                return biomarker
        return None

    def _contains_keyword(self, normalized_text: str, keyword: str) -> bool:
        """Match keyword an toàn hơn cho acronym ngắn như AKI, CKD, GFR."""

        if " " in keyword or len(keyword) > 4:
            return keyword in normalized_text
        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        return re.search(pattern, normalized_text) is not None

    def _rfind_keyword(self, normalized_text: str, keyword: str) -> int:
        """Tìm vị trí xuất hiện cuối cùng của keyword với boundary an toàn."""

        if " " in keyword or len(keyword) > 4:
            return normalized_text.rfind(keyword)

        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        matches = list(re.finditer(pattern, normalized_text))
        if not matches:
            return -1
        return matches[-1].start()

    def _parse_rule_condition(self, condition: str) -> tuple[float | None, float | None, str]:
        cleaned = self._normalize_ascii(condition).replace(",", ".")

        between_match = re.search(r"(\d+(?:\.\d+)?)\s*<=\s*[a-z_]+\s*<=\s*(\d+(?:\.\d+)?)", cleaned)
        if between_match:
            return float(between_match.group(1)), float(between_match.group(2)), "between"

        ge_lt_match = re.search(r"(\d+(?:\.\d+)?)\s*<=\s*[a-z_]+\s*<\s*(\d+(?:\.\d+)?)", cleaned)
        if ge_lt_match:
            return float(ge_lt_match.group(1)), float(ge_lt_match.group(2)), "between"

        single_match = re.search(r"[a-z_]+\s*(<=|>=|<|>)\s*(\d+(?:\.\d+)?)", cleaned)
        if single_match:
            value = float(single_match.group(2))
            return value, None, single_match.group(1)

        return None, None, "unknown"

    def _infer_threshold_unit(self, text: str, biomarker: str) -> str | None:
        for match in INLINE_VALUE_RE.finditer(text):
            context_before = self._normalize_ascii(text[: match.start()])
            detected = self._find_last_biomarker_before(context_before)
            if detected == biomarker:
                return self._normalize_unit(match.group("unit"))
        return THRESHOLD_UNIT_DEFAULTS.get(biomarker)

    def _infer_formula_disease_name(self, formula_name: str | None, text: str) -> str | None:
        """Gán disease fallback cho các công thức không nhắc rõ bệnh trong cùng đoạn text."""

        detected = self._detect_disease_name(text)
        if detected:
            return detected

        normalized_name = self._normalize_ascii(formula_name or "")
        if any(token in normalized_name for token in ("mdrd", "cockcroft", "dien tich da", "bsa")):
            return "benh_than_man"
        if "fena" in normalized_name:
            return "acute_kidney_injury"
        return None

    def _build_page_disease_context(self, chunks: list[dict[str, Any]]) -> dict[int, str]:
        """Rút ra disease context theo trang từ các chunk đã được propagate context."""

        context: dict[int, str] = {}
        for item in chunks:
            disease_name = item["metadata"].get("disease_name")
            if disease_name is None:
                continue
            context.setdefault(item["metadata"]["page"], disease_name)
        return context

    def _resolve_disease_name(
        self,
        text: str,
        explicit_disease: str | None,
        current_disease_context: str | None,
    ) -> str | None:
        """
        Quyết định disease_name cuối cùng cho block hiện tại.

        Nếu block chỉ nhắc một bệnh phụ trong khi đang đứng trong một chapter bệnh chính,
        ưu tiên giữ context bệnh chính thay vì để bệnh phụ cướp metadata của cả block.
        """

        if current_disease_context and not self._is_disease_anchor_text(text, explicit_disease):
            return current_disease_context if explicit_disease is None or explicit_disease != current_disease_context else explicit_disease
        return explicit_disease or current_disease_context

    def _is_context_reset_text(self, text: str) -> bool:
        """Các đoạn overview lớn sẽ reset disease context để tránh kéo sai sang phần khác."""

        normalized = self._normalize_ascii(text)
        return any(marker in normalized for marker in CONTEXT_RESET_MARKERS)

    def _is_disease_anchor_text(self, text: str, explicit_disease: str | None) -> bool:
        """
        Chỉ những đoạn giống tiêu đề/chương bệnh mới được phép mở context kéo dài.

        Điều này giúp tránh việc một subtype được nhắc thoáng qua làm đổi disease context
        của cả đoạn hoặc cả trang.
        """

        if explicit_disease is None:
            return False

        heading_line = self._extract_heading_line(text)
        heading_level = self._extract_heading_level(text)
        normalized_text = self._normalize_ascii(text[:300])
        normalized_heading = self._normalize_ascii(heading_line)

        if self._looks_like_uppercase_title(heading_line):
            return True
        if heading_level == 1 and explicit_disease in self._normalize_ascii(text):
            return True
        if ":" in heading_line and explicit_disease in normalized_text:
            return True
        if heading_level == 1 and explicit_disease in normalized_heading:
            return True
        return False

    def _extract_heading_line(self, text: str) -> str:
        """Lấy dòng đầu tiên có nội dung để phục vụ suy luận heading/context."""

        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return ""

    def _extract_heading_number(self, text: str) -> str | None:
        """Lấy phần số của heading, ví dụ `4.4.8.1`."""

        heading_line = self._extract_heading_line(text)
        match = HEADING_PREFIX_RE.match(heading_line)
        if not match:
            return None
        return match.group("number")

    def _extract_heading_level(self, text: str) -> int | None:
        """Suy ra cấp heading từ dòng đầu tiên, ví dụ `4.1.2` -> level 3."""

        heading_number = self._extract_heading_number(text)
        if heading_number is None:
            return None
        return len(heading_number.split("."))

    def _looks_like_uppercase_title(self, text: str) -> bool:
        """Nhận diện tiêu đề kiểu `BỆNH THẬN LUPUS` không có số thứ tự."""

        stripped = text.strip()
        if not stripped or len(stripped) > 120:
            return False
        letters = [char for char in stripped if char.isalpha()]
        if not letters:
            return False
        uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
        return uppercase_ratio > 0.7

    def _make_chunk_id(self, disease_name: str | None, page: int, chunk_index: int) -> str:
        disease_token = disease_name or "unknown"
        return f"{disease_token}_p{page}_{chunk_index:03d}"

    def _make_threshold_id(
        self,
        disease_name: str | None,
        biomarker: str,
        operator: str,
        value: str,
        page: int,
        suffix: int,
    ) -> str:
        disease_token = disease_name or "unknown"
        op_token = operator.replace("<", "lt").replace(">", "gt").replace("=", "eq")
        value_token = value.replace(".", "_").replace(",", "_")
        return f"{disease_token}_{self._slugify(biomarker)}_{op_token}_{value_token}_p{page}_{suffix:03d}"

    def _slugify(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()

    def _normalize_ascii(self, value: str) -> str:
        value = value.replace("đ", "d").replace("Đ", "D")
        normalized = unicodedata.normalize("NFKD", value)
        ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
        return re.sub(r"\s+", " ", ascii_value)

    def _normalize_operator(self, value: str) -> str:
        lowered = self._normalize_ascii(value)
        mapping = {
            "tren": ">",
            "cao tren": ">",
            "tang tren": ">",
            "duoi": "<",
            "giam duoi": "<",
        }
        return mapping.get(lowered, lowered)

    def _normalize_unit(self, value: str) -> str:
        lowered = value.strip()
        return (
            lowered.replace("g/24 giờ", "g/24h")
            .replace("g/24 gio", "g/24h")
            .replace("ml/phút/1.73m2", "ml/ph/1.73m2")
            .replace("ml/phút/1,73m2", "ml/ph/1.73m2")
            .replace("ml/ph/1,73m2", "ml/ph/1.73m2")
            .replace("g/l", "g/L")
            .replace("mmol/l", "mmol/L")
        )

    def _clean_json_string(self, value: str) -> str:
        text = value.strip()
        if not text:
            return ""

        if "\\u" in text or "\\n" in text or "\\t" in text or '\\"' in text or "\\/" in text:
            try:
                text = json.loads(f'"{text}"')
            except json.JSONDecodeError:
                text = text.replace("\\n", "\n").replace("\\t", " ").replace('\\"', '"').replace("\\/", "/")

        return " ".join(segment.strip() for segment in text.splitlines() if segment.strip())

    def _to_float(self, value: str) -> float:
        return float(value.replace(",", "."))

    def _dedupe_by_key(self, items: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
        """Loại duplicate theo id ổn định để output nhất quán giữa các lần chạy."""

        seen = set()
        unique: list[dict[str, Any]] = []

        for item in items:
            key = item.get(key_name)
            payload = json.dumps(item, ensure_ascii=False, sort_keys=True)
            dedupe_key = key if key else payload
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            unique.append(item)

        return unique

    def _build_summary(
        self,
        pages: list[PageRecord],
        chunks: list[dict[str, Any]],
        thresholds: list[dict[str, Any]],
        formulas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Tạo summary ngắn để nhìn nhanh chất lượng extraction sau mỗi lần chạy."""

        return {
            "source_file": self.source_file,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "threshold_count": len(thresholds),
            "formula_count": len(formulas),
            "chunk_content_types": dict(Counter(item["metadata"]["content_type"] for item in chunks)),
            "threshold_biomarkers": dict(Counter(item["biomarker"] for item in thresholds)),
            "formula_names": sorted({item["formula_name"] for item in formulas}),
        }

    def write_outputs(self, output_dir: str | Path) -> dict[str, Path]:
        """Ghi toàn bộ artifact đã extract xuống thư mục output."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = self.process()
        paths = {
            "chunks": output_path / "chunks.jsonl",
            "thresholds": output_path / "thresholds.jsonl",
            "formulas": output_path / "formulas.json",
            "summary": output_path / "summary.json",
        }

        self._write_jsonl(paths["chunks"], results["chunks"])
        self._write_jsonl(paths["thresholds"], results["thresholds"])
        paths["formulas"].write_text(json.dumps(results["formulas"], ensure_ascii=False, indent=2), encoding="utf-8")
        paths["summary"].write_text(json.dumps(results["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
        return paths

    def _write_jsonl(self, path: Path, items: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
