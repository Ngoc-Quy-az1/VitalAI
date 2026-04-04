from __future__ import annotations

"""
Retriever hybrid để test retrieval trực tiếp trên Neon + pgvector.

File này dành cho giai đoạn kiểm tra chất lượng retrieval:
- hiểu query ở mức heuristic
- embed query đã được enrich theo intent
- kết hợp vector search + full-text search
- cộng bonus theo metadata hint để giảm retrieval noise

Chưa làm ở bước này:
- reranker riêng bằng model
- structured lookup riêng cho threshold/formula
- query understanding dùng model
"""

import json
import os
import re
import unicodedata
from typing import Any

import asyncpg
from dotenv import load_dotenv
from openai import AsyncOpenAI

DISEASE_HINTS = {
    "benh_than_man": ["benh than man", "suy than man", "ckd", "chronic kidney disease"],
    "lupus_nephritis": ["lupus", "benh than lupus", "viem than lupus", "lupus ban do"],
    "acute_kidney_injury": ["suy than cap", "aki", "ton thuong than cap"],
    "hoi_chung_than_hu": ["hoi chung than hu"],
    "benh_than_iga": ["iga", "benh than iga"],
    "diabetic_kidney_disease": ["dai thao duong", "than dai thao duong", "diabetic kidney disease"],
}

SECTION_HINTS = {
    "definition": ["la gi", "khai niem", "dinh nghia", "co nghia la gi"],
    "classification": ["phan loai", "giai doan", "stage", "kdigo", "a1", "a2", "a3", "g1", "g2", "g3", "g4", "g5"],
    "diagnosis_criteria": ["chan doan", "tieu chuan chan doan"],
    "treatment": ["dieu tri", "thuoc", "phac do"],
    "clinical_features": ["trieu chung", "lam sang", "can lam sang", "bieu hien"],
    "pathology": ["mo benh hoc", "sinh thiet", "mien dich huynh quang"],
    "progression": ["tien trien", "tien luong"],
    "follow_up": ["theo doi", "tai kham", "du phong", "phong ngua"],
}

BIOMARKER_HINTS = {
    "ACR": ["acr", "albumin creatinine ratio", "albumin/creatinin", "albumin/creatinine"],
    "PCR": ["pcr", "protein/creatinin", "protein/creatinine", "protein niu/creatinin"],
    "GFR": ["gfr", "egfr", "muc loc cau than", "mlct"],
    "creatinine": ["creatinin", "creatinine"],
    "protein_niệu_24h": ["protein nieu 24h", "protein nieu", "protein niu 24h"],
}

DISEASE_LABELS = {
    "benh_than_man": "Bệnh thận mạn",
    "lupus_nephritis": "Viêm thận lupus",
    "acute_kidney_injury": "Tổn thương thận cấp",
    "hoi_chung_than_hu": "Hội chứng thận hư",
    "benh_than_iga": "Bệnh thận IgA",
    "diabetic_kidney_disease": "Bệnh thận do đái tháo đường",
}

SECTION_LABELS = {
    "definition": "Khái niệm",
    "classification": "Phân loại",
    "diagnosis_criteria": "Chẩn đoán",
    "treatment": "Điều trị",
    "clinical_features": "Lâm sàng và cận lâm sàng",
    "pathology": "Mô bệnh học",
    "progression": "Tiến triển và tiên lượng",
    "follow_up": "Theo dõi",
}


class NeonVectorSearcher:
    """Thực hiện vector search trên bảng `medical_documents`."""

    def __init__(
        self,
        database_url: str,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
    ) -> None:
        self.database_url = database_url
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.openai = AsyncOpenAI(api_key=openai_api_key)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        disease_name: str | None = None,
        section_type: str | None = None,
        source_type: str | None = None,
        biomarker: str | None = None,
    ) -> dict[str, Any]:
        """Thực hiện hybrid retrieval và trả về top-k kết quả tốt nhất."""

        understanding = self._understand_query(
            query=query,
            disease_name=disease_name,
            section_type=section_type,
            biomarker=biomarker,
        )
        query_embedding = await self._embed_query(understanding["embedding_query"])
        conn = await asyncpg.connect(self.database_url, statement_cache_size=0)
        try:
            candidate_limit = max(top_k * 6, 12)
            vector_rows = await self._search_vector_rows(
                conn=conn,
                embedding=query_embedding,
                top_k=candidate_limit,
                disease_name=disease_name,
                section_type=section_type,
                source_type=source_type,
                biomarker=biomarker,
            )
            keyword_rows = await self._search_keyword_rows(
                conn=conn,
                keyword_query=understanding["keyword_query"],
                top_k=candidate_limit,
                disease_name=disease_name,
                section_type=section_type,
                source_type=source_type,
                biomarker=biomarker,
            )
            rows = self._fuse_rows(
                vector_rows=vector_rows,
                keyword_rows=keyword_rows,
                top_k=top_k,
                understanding=understanding,
            )
        finally:
            await conn.close()

        return {
            "query": query,
            "top_k": top_k,
            "strategy": "hybrid_vector_fts_rrf",
            "filters": {
                "disease_name": disease_name,
                "section_type": section_type,
                "source_type": source_type,
                "biomarker": biomarker,
            },
            "query_understanding": {
                "disease_hint": understanding["disease_hint"],
                "section_hint": understanding["section_hint"],
                "biomarker_hint": understanding["biomarker_hint"],
                "embedding_query": understanding["embedding_query"],
                "keyword_query": understanding["keyword_query"],
            },
            "results": rows,
        }

    async def _embed_query(self, query: str) -> list[float]:
        """Gọi OpenAI Embeddings cho câu truy vấn."""

        response = await self.openai.embeddings.create(
            model=self.embedding_model,
            input=[query],
            dimensions=self.embedding_dimensions,
        )
        return response.data[0].embedding

    async def _search_vector_rows(
        self,
        conn: asyncpg.Connection,
        embedding: list[float],
        top_k: int,
        disease_name: str | None,
        section_type: str | None,
        source_type: str | None,
        biomarker: str | None,
    ) -> list[dict[str, Any]]:
        """Lấy candidate theo vector similarity."""

        filters: list[str] = []
        params: list[Any] = [json.dumps(embedding)]

        self._append_filters(
            params=params,
            filters=filters,
            disease_name=disease_name,
            section_type=section_type,
            source_type=source_type,
            biomarker=biomarker,
        )

        params.append(top_k)
        limit_index = len(params)

        where_clause = ""
        if filters:
            where_clause = "WHERE " + " AND ".join(filters)

        sql = f"""
        SELECT
            document_id,
            source_type,
            source_id,
            content,
            metadata,
            1 - (embedding <=> $1::vector) AS similarity,
            NULL::float8 AS keyword_score
        FROM medical_documents
        {where_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT ${limit_index}
        """

        rows = await conn.fetch(sql, *params)
        return [self._normalize_result_row(row) for row in rows]

    async def _search_keyword_rows(
        self,
        conn: asyncpg.Connection,
        keyword_query: str,
        top_k: int,
        disease_name: str | None,
        section_type: str | None,
        source_type: str | None,
        biomarker: str | None,
    ) -> list[dict[str, Any]]:
        """Lấy candidate theo full-text search trên content."""

        if not keyword_query.strip():
            return []

        filters: list[str] = ["websearch_to_tsquery('simple', $1) @@ to_tsvector('simple', content)"]
        params: list[Any] = [keyword_query]
        self._append_filters(
            params=params,
            filters=filters,
            disease_name=disease_name,
            section_type=section_type,
            source_type=source_type,
            biomarker=biomarker,
        )

        params.append(top_k)
        limit_index = len(params)
        where_clause = "WHERE " + " AND ".join(filters)

        sql = f"""
        SELECT
            document_id,
            source_type,
            source_id,
            content,
            metadata,
            NULL::float8 AS similarity,
            ts_rank_cd(
                to_tsvector('simple', content),
                websearch_to_tsquery('simple', $1)
            ) AS keyword_score
        FROM medical_documents
        {where_clause}
        ORDER BY keyword_score DESC, document_id ASC
        LIMIT ${limit_index}
        """

        rows = await conn.fetch(sql, *params)
        return [self._normalize_result_row(row) for row in rows]

    def _append_filters(
        self,
        params: list[Any],
        filters: list[str],
        disease_name: str | None,
        section_type: str | None,
        source_type: str | None,
        biomarker: str | None,
    ) -> None:
        """Dùng chung logic build WHERE cho vector và FTS query."""

        if disease_name:
            params.append(disease_name)
            filters.append(f"metadata->>'disease_name' = ${len(params)}")
        if section_type:
            params.append(section_type)
            filters.append(f"metadata->>'section_type' = ${len(params)}")
        if source_type:
            params.append(source_type)
            filters.append(f"source_type = ${len(params)}")
        if biomarker:
            params.append(biomarker)
            filters.append(f"metadata->>'biomarker' = ${len(params)}")

    def _normalize_result_row(self, row: asyncpg.Record) -> dict[str, Any]:
        """Chuẩn hóa một row DB thành dict nhất quán để dễ fusion."""

        metadata = row["metadata"] or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        content = row["content"]
        similarity = row["similarity"]
        keyword_score = row["keyword_score"]
        return {
            "document_id": row["document_id"],
            "source_type": row["source_type"],
            "source_id": row["source_id"],
            "content": content,
            "similarity": round(float(similarity), 6) if similarity is not None else None,
            "keyword_score": round(float(keyword_score), 6) if keyword_score is not None else None,
            "page": metadata.get("page"),
            "disease_name": metadata.get("disease_name"),
            "section_type": metadata.get("section_type"),
            "doc_type": metadata.get("doc_type"),
            "biomarker": metadata.get("biomarker"),
            "preview": content[:500],
        }

    def _fuse_rows(
        self,
        vector_rows: list[dict[str, Any]],
        keyword_rows: list[dict[str, Any]],
        top_k: int,
        understanding: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Hợp nhất candidate theo RRF + bonus metadata phù hợp intent."""

        fused: dict[str, dict[str, Any]] = {}

        for rank, row in enumerate(vector_rows, start=1):
            item = fused.setdefault(row["document_id"], dict(row))
            item["vector_rank"] = rank
            item["rrf_score"] = item.get("rrf_score", 0.0) + (1.0 / (60 + rank))

        for rank, row in enumerate(keyword_rows, start=1):
            item = fused.setdefault(row["document_id"], dict(row))
            item["keyword_rank"] = rank
            item["rrf_score"] = item.get("rrf_score", 0.0) + (1.0 / (60 + rank))
            if item.get("similarity") is None:
                item["similarity"] = row.get("similarity")
            if item.get("keyword_score") is None:
                item["keyword_score"] = row.get("keyword_score")
            if item.get("preview") is None:
                item["preview"] = row.get("preview")
            if item.get("content") is None:
                item["content"] = row.get("content")

        for item in fused.values():
            item["metadata_bonus"] = self._metadata_bonus(item, understanding)
            item["fusion_score"] = round(item.get("rrf_score", 0.0) + item["metadata_bonus"], 6)

        ranked = sorted(
            fused.values(),
            key=lambda item: (
                item["fusion_score"],
                item.get("keyword_score") or 0.0,
                item.get("similarity") or 0.0,
            ),
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        for item in ranked[:top_k]:
            results.append(
                {
                    "document_id": item["document_id"],
                    "source_type": item["source_type"],
                    "source_id": item["source_id"],
                    "similarity": item.get("similarity"),
                    "keyword_score": item.get("keyword_score"),
                    "fusion_score": item["fusion_score"],
                    "page": item.get("page"),
                    "disease_name": item.get("disease_name"),
                    "section_type": item.get("section_type"),
                    "doc_type": item.get("doc_type"),
                    "biomarker": item.get("biomarker"),
                    "vector_rank": item.get("vector_rank"),
                    "keyword_rank": item.get("keyword_rank"),
                    "preview": item.get("preview", ""),
                }
            )
        return results

    def _metadata_bonus(self, row: dict[str, Any], understanding: dict[str, Any]) -> float:
        """Cộng điểm mềm cho document khớp disease/section/biomarker hint."""

        bonus = 0.0

        if understanding["disease_hint"] and row.get("disease_name") == understanding["disease_hint"]:
            bonus += 0.04
        if understanding["section_hint"] and row.get("section_type") == understanding["section_hint"]:
            bonus += 0.08
        if understanding["section_hint"] == "definition" and row.get("section_type") == "general":
            bonus += 0.03
        if understanding["section_hint"] == "definition" and row.get("source_type") == "chunk":
            bonus += 0.01
        if understanding["biomarker_hint"] and row.get("biomarker") == understanding["biomarker_hint"]:
            bonus += 0.03

        return round(bonus, 6)

    def _understand_query(
        self,
        query: str,
        disease_name: str | None,
        section_type: str | None,
        biomarker: str | None,
    ) -> dict[str, Any]:
        """Suy ra intent retrieval cơ bản từ query để enrich search."""

        normalized = self._normalize_ascii(query)
        disease_hint = disease_name or self._detect_hint(normalized, DISEASE_HINTS)
        section_hint = section_type or self._detect_hint(normalized, SECTION_HINTS)
        biomarker_hint = biomarker or self._detect_hint(normalized, BIOMARKER_HINTS)

        if section_hint is None and any(pattern in normalized for pattern in ("la gi", "khai niem", "dinh nghia")):
            section_hint = "definition"

        embedding_lines = [f"Câu hỏi người dùng: {query.strip()}"]
        if disease_hint:
            embedding_lines.append(f"Bệnh trọng tâm: {DISEASE_LABELS.get(disease_hint, disease_hint)}")
        if section_hint:
            embedding_lines.append(f"Mục cần tìm: {SECTION_LABELS.get(section_hint, section_hint)}")
        if biomarker_hint:
            embedding_lines.append(f"Chỉ số trọng tâm: {biomarker_hint}")

        keyword_query = query.strip()
        if section_hint == "definition" and "khai niem" not in normalized and "dinh nghia" not in normalized:
            keyword_query = f"{keyword_query} khái niệm định nghĩa"

        return {
            "disease_hint": disease_hint,
            "section_hint": section_hint,
            "biomarker_hint": biomarker_hint,
            "embedding_query": "\n".join(embedding_lines),
            "keyword_query": keyword_query,
        }

    def _detect_hint(self, normalized_query: str, hint_map: dict[str, list[str]]) -> str | None:
        """Tìm hint đầu tiên khớp query theo danh sách alias đơn giản."""

        for value, aliases in hint_map.items():
            if any(self._contains_keyword(normalized_query, alias) for alias in aliases):
                return value
        return None

    def _contains_keyword(self, normalized_text: str, keyword: str) -> bool:
        """Match an toàn hơn cho acronym ngắn như ACR, GFR, CKD."""

        if " " in keyword or len(keyword) > 4:
            return keyword in normalized_text
        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        return re.search(pattern, normalized_text) is not None

    def _normalize_ascii(self, text: str) -> str:
        """Bỏ dấu tiếng Việt và co khoảng trắng để match heuristic ổn định hơn."""

        normalized = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
        return " ".join(ascii_text.split())


def build_searcher_from_env() -> NeonVectorSearcher:
    """Khởi tạo searcher từ `.env` hoặc env hiện tại."""

    load_dotenv()

    database_url = os.getenv("NEON_DATABASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

    if not database_url:
        raise ValueError("Thiếu biến môi trường NEON_DATABASE_URL")
    if not openai_api_key:
        raise ValueError("Thiếu biến môi trường OPENAI_API_KEY")

    return NeonVectorSearcher(
        database_url=database_url,
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )
