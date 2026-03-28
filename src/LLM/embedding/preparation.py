from __future__ import annotations

"""
Bộ chuẩn bị dữ liệu embedding cho tài liệu y khoa hiện tại.

File này tồn tại để tách rõ 2 giai đoạn:
1. `processed_data`: dữ liệu sau extraction, còn mang tính debug / QA.
2. `embedding_data`: dữ liệu đã được chọn lọc, diễn đạt lại và sẵn sàng đưa vào vector DB.

Nguyên tắc ở bước này:
- chỉ giữ những document thật sự có ích cho retrieval
- loại các chunk nhiễu như heading trống, bullet rời, text quá ngắn
- tạo thêm companion document cho threshold và formula để retrieval semantic có thêm ngữ cảnh
- không gọi API embedding, không ghi database
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DISEASE_LABELS = {
    "benh_ly_cau_than": "Bệnh lý cầu thận",
    "benh_than_man": "Bệnh thận mạn",
    "lupus_nephritis": "Viêm thận lupus",
    "acute_kidney_injury": "Tổn thương thận cấp",
    "hoi_chung_than_hu": "Hội chứng thận hư",
    "viem_cau_than_cap": "Viêm cầu thận cấp",
    "viem_cau_than_man": "Viêm cầu thận mạn",
    "viem_cau_than_tien_trien_nhanh": "Viêm cầu thận tiến triển nhanh",
    "benh_than_iga": "Bệnh thận IgA",
    "diabetic_kidney_disease": "Bệnh thận do đái tháo đường",
}

SECTION_LABELS = {
    "definition": "Khái niệm",
    "classification": "Phân loại",
    "clinical_features": "Lâm sàng và cận lâm sàng",
    "diagnosis_criteria": "Chẩn đoán",
    "pathology": "Mô bệnh học",
    "treatment": "Điều trị",
    "progression": "Tiến triển và tiên lượng",
    "complications": "Biến chứng",
    "follow_up": "Theo dõi và dự phòng",
    "general": "Tổng quát",
}

DOC_TYPE_LABELS = {
    "disease_guideline": "Hướng dẫn bệnh học",
    "threshold_reference": "Ngưỡng tham chiếu",
    "formula_reference": "Công thức tham chiếu",
    "medication_reference": "Thông tin điều trị",
}

BIOMARKER_LABELS = {
    "protein_niệu_24h": "Protein niệu 24 giờ",
    "protein_mau": "Protein máu",
    "albumin_máu": "Albumin máu",
    "cholesterol": "Cholesterol",
    "creatinine": "Creatinine",
    "PCR": "Tỷ số protein/creatinin niệu",
    "ACR": "Tỷ số albumin/creatinin niệu",
    "GFR": "Mức lọc cầu thận",
    "BSA": "Diện tích da cơ thể",
    "FENa": "Fractional Excretion of Sodium",
    "sodium": "Natri",
    "urea": "Ure",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Đọc một file JSONL UTF-8 thành list dict."""

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class EmbeddingDataPreparer:
    """
    Tạo artifact embedding-ready từ dữ liệu đã extract.

    Kết quả đầu ra chính:
    - `embedding_documents.jsonl`: danh sách document sẽ được embed
    - `embedding_manifest.json`: thống kê và QA nhanh cho bước chuẩn bị embedding
    """

    def __init__(self, processed_dir: str | Path) -> None:
        self.processed_dir = Path(processed_dir)

    def prepare(self) -> dict[str, Any]:
        """Đọc artifact đã extract, chọn lọc và biến chúng thành embedding documents."""

        chunks = load_jsonl(self.processed_dir / "chunks.jsonl")
        thresholds = load_jsonl(self.processed_dir / "thresholds.jsonl")
        formulas = json.loads((self.processed_dir / "formulas.json").read_text(encoding="utf-8"))

        documents: list[dict[str, Any]] = []
        skipped_chunks: list[dict[str, Any]] = []

        for index, chunk in enumerate(chunks):
            prepared = self._build_chunk_document(chunks, index)
            if prepared is None:
                skipped_chunks.append(
                    {
                        "source_id": chunk["chunk_id"],
                        "reason": "chunk_nhieu_nhieu_hoac_qua_ngan",
                        "page": chunk["metadata"]["page"],
                        "preview": chunk["content"][:160],
                    }
                )
                continue
            documents.append(prepared)

        for threshold in thresholds:
            documents.append(self._build_threshold_document(threshold))

        for formula in formulas:
            documents.append(self._build_formula_document(formula))

        unique_documents = self._dedupe_documents(documents)
        manifest = self._build_manifest(unique_documents, skipped_chunks)

        return {
            "documents": unique_documents,
            "manifest": manifest,
        }

    def write_outputs(self, output_dir: str | Path) -> dict[str, Path]:
        """Ghi artifact embedding-ready xuống thư mục output."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = self.prepare()
        paths = {
            "documents": output_path / "embedding_documents.jsonl",
            "manifest": output_path / "embedding_manifest.json",
        }
        self._write_jsonl(paths["documents"], results["documents"])
        paths["manifest"].write_text(json.dumps(results["manifest"], ensure_ascii=False, indent=2), encoding="utf-8")
        return paths

    def _build_chunk_document(self, chunks: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
        """Biến một chunk prose/json thành document sẽ đi embed."""

        chunk = dict(chunks[index])
        chunk["metadata"] = dict(chunk["metadata"])
        if chunk["metadata"].get("disease_name") is None:
            chunk["metadata"]["disease_name"] = self._infer_chunk_disease_from_neighbors(chunks, index)

        content = self._normalize_whitespace(chunk["content"])
        if self._is_noise_chunk(content):
            return None

        metadata = chunk["metadata"]
        header_lines = [
            f"Bệnh: {self._humanize_disease(metadata.get('disease_name'))}",
            f"Mục: {self._humanize_section(metadata.get('section_type'))}",
            f"Loại tài liệu: {self._humanize_doc_type(metadata.get('doc_type'))}",
        ]
        if metadata.get("biomarker"):
            header_lines.append(f"Chỉ số liên quan: {self._humanize_biomarker(metadata['biomarker'])}")

        embedding_text = "\n".join(header_lines + ["Nội dung:", content])

        return {
            "document_id": f"chunk::{chunk['chunk_id']}",
            "source_type": "chunk",
            "source_id": chunk["chunk_id"],
            "content": content,
            "embedding_text": embedding_text,
            "metadata": metadata,
        }

    def _infer_chunk_disease_from_neighbors(self, chunks: list[dict[str, Any]], current_index: int) -> str | None:
        """
        Suy luận disease cho chunk còn thiếu metadata bằng các chunk lân cận.

        Bước này chỉ dùng ở artifact embedding-ready để tránh một vài document tốt
        bị bỏ sót chỉ vì heading tổng quát chưa được gán disease ở phase extraction.
        """

        current_page = chunks[current_index]["metadata"]["page"]
        candidates: list[str] = []

        for offset in range(1, 6):
            for neighbor_index in (current_index - offset, current_index + offset):
                if neighbor_index < 0 or neighbor_index >= len(chunks):
                    continue
                neighbor = chunks[neighbor_index]
                disease_name = neighbor["metadata"].get("disease_name")
                if disease_name is None:
                    continue
                if abs(neighbor["metadata"]["page"] - current_page) > 1:
                    continue
                candidates.append(disease_name)

        if not candidates:
            return None

        counts = Counter(candidates).most_common()
        if len(counts) == 1:
            return counts[0][0]
        if counts[0][1] > counts[1][1]:
            return counts[0][0]
        return None

    def _build_threshold_document(self, threshold: dict[str, Any]) -> dict[str, Any]:
        """Tạo companion document dạng text cho một threshold structured."""

        disease_label = self._humanize_disease(threshold.get("disease_name"))
        biomarker_label = self._humanize_biomarker(threshold.get("biomarker"))
        op_label = self._humanize_operator(threshold.get("threshold_op"))
        value_text = self._format_threshold_value(threshold)
        label_text = f" Phân loại: {threshold['label']}." if threshold.get("label") else ""

        content = (
            f"Trong bối cảnh {disease_label}, chỉ số {biomarker_label} có ngưỡng {op_label} {value_text}."
            f"{label_text} Trích từ trang {threshold['page']}."
        ).strip()

        return {
            "document_id": f"threshold::{threshold['threshold_id']}",
            "source_type": "threshold",
            "source_id": threshold["threshold_id"],
            "content": content,
            "embedding_text": content,
            "metadata": {
                "doc_type": "threshold_reference",
                "disease_name": threshold.get("disease_name"),
                "section_type": threshold.get("section_type"),
                "content_type": "threshold_companion",
                "biomarker": threshold.get("biomarker"),
                "source_file": threshold.get("source_file"),
                "page": threshold.get("page"),
                "language": threshold.get("language", "vi"),
            },
        }

    def _build_formula_document(self, formula: dict[str, Any]) -> dict[str, Any]:
        """Tạo companion document dạng text cho một công thức structured."""

        disease_label = self._humanize_disease(formula.get("disease_name"))
        variables = ", ".join(item["name"] for item in formula.get("variables", [])) or "không rõ biến đầu vào"
        output_name = formula.get("output_name") or "không rõ đầu ra"
        output_unit = f" ({formula['output_unit']})" if formula.get("output_unit") else ""

        content = (
            f"Công thức {formula['formula_name']} áp dụng trong bối cảnh {disease_label}. "
            f"Biểu thức: {formula['expression']}. "
            f"Biến đầu vào: {variables}. "
            f"Đầu ra: {output_name}{output_unit}. "
            f"Trích từ trang {formula['page']}."
        )

        return {
            "document_id": f"formula::{formula['formula_id']}",
            "source_type": "formula",
            "source_id": formula["formula_id"],
            "content": content,
            "embedding_text": content,
            "metadata": {
                "doc_type": "formula_reference",
                "disease_name": formula.get("disease_name"),
                "section_type": formula.get("section_type"),
                "content_type": "formula_companion",
                "biomarker": None,
                "source_file": formula.get("source_file"),
                "page": formula.get("page"),
                "language": formula.get("language", "vi"),
            },
        }

    def _is_noise_chunk(self, content: str) -> bool:
        """Loại các chunk quá ngắn hoặc gần như chỉ là heading / bullet."""

        compact = content.strip()
        if len(compact) < 30:
            return True
        if re.fullmatch(r"[\W_•]+", compact):
            return True

        letters = [char for char in compact if char.isalpha()]
        if len(letters) < 18:
            return True

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) == 1 and len(lines[0]) < 45:
            return True
        return False

    def _build_manifest(self, documents: list[dict[str, Any]], skipped_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Tạo manifest để nhìn nhanh chất lượng dữ liệu trước khi embed thật."""

        return {
            "counts": {
                "documents": len(documents),
                "skipped_chunks": len(skipped_chunks),
            },
            "distributions": {
                "source_type": dict(Counter(item["source_type"] for item in documents)),
                "doc_type": dict(Counter(item["metadata"]["doc_type"] for item in documents)),
                "section_type": dict(Counter(item["metadata"]["section_type"] for item in documents)),
                "disease_name": dict(Counter(item["metadata"].get("disease_name") for item in documents)),
            },
            "samples": {
                "documents": documents[:5],
                "skipped_chunks": skipped_chunks[:10],
            },
        }

    def _dedupe_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Khử trùng lặp theo `document_id` và theo nội dung embedding đã normalize."""

        unique: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()

        for item in documents:
            document_id = item["document_id"]
            normalized_text = self._normalize_whitespace(item["embedding_text"]).lower()
            if document_id in seen_ids or normalized_text in seen_texts:
                continue
            seen_ids.add(document_id)
            seen_texts.add(normalized_text)
            unique.append(item)

        return unique

    def _normalize_whitespace(self, text: str) -> str:
        """Co khoảng trắng để nội dung gọn hơn trước khi embed."""

        normalized = text.replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _humanize_disease(self, disease_name: str | None) -> str:
        return DISEASE_LABELS.get(disease_name or "", disease_name or "không rõ bệnh")

    def _humanize_section(self, section_type: str | None) -> str:
        return SECTION_LABELS.get(section_type or "", section_type or "không rõ mục")

    def _humanize_doc_type(self, doc_type: str | None) -> str:
        return DOC_TYPE_LABELS.get(doc_type or "", doc_type or "không rõ loại")

    def _humanize_biomarker(self, biomarker: str | None) -> str:
        return BIOMARKER_LABELS.get(biomarker or "", biomarker or "không rõ chỉ số")

    def _humanize_operator(self, operator: str | None) -> str:
        mapping = {
            ">": "lớn hơn",
            "<": "nhỏ hơn",
            ">=": "lớn hơn hoặc bằng",
            "<=": "nhỏ hơn hoặc bằng",
            "between": "nằm trong khoảng",
        }
        return mapping.get(operator or "", operator or "không rõ toán tử")

    def _format_threshold_value(self, threshold: dict[str, Any]) -> str:
        """Format threshold value theo kiểu dễ đọc cho companion document."""

        if threshold.get("threshold_op") == "between":
            min_value = threshold.get("threshold_value_min")
            max_value = threshold.get("threshold_value_max")
            unit = threshold.get("threshold_unit") or ""
            return f"{min_value} đến {max_value} {unit}".strip()

        value = threshold.get("threshold_value")
        unit = threshold.get("threshold_unit") or ""
        return f"{value} {unit}".strip()

    def _write_jsonl(self, path: Path, items: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
