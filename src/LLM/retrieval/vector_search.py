from __future__ import annotations


import json
import os
import re
import asyncio
import unicodedata
from typing import Any

import asyncpg
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .bm25_retriever import BM25Retriever, default_bm25_index_path
from .reranker import build_reranker, LocalReranker, NullReranker

MAX_RESULT_CONTEXT_CHARS = 1800
NEIGHBOR_BACKWARD_WINDOW = 1
NEIGHBOR_FORWARD_WINDOW = 4
RRF_K = 60
VECTOR_RRF_WEIGHT = 0.85
KEYWORD_RRF_WEIGHT = 1.0
LEXICAL_RRF_WEIGHT = 1.15

DISEASE_HINTS = {
    "lupus_nephritis": ["lupus", "benh than lupus", "viem than lupus", "lupus ban do", "ara 1997", "class iv"],
    "benh_than_iga": ["iga", "benh than iga", "viem cau than iga", "berger"],
    "hoi_chung_than_hu": ["hoi chung than hu", "than hu", "nephrotic syndrome"],
    "viem_cau_than_cap": ["viem cau than cap", "cap sau nhiem lien cau", "sau nhiem lien cau", "psagn"],
    "acute_kidney_injury": ["suy than cap", "aki", "ton thuong than cap", "rifle", "fena", "kdigo 2012"],
    "diabetic_kidney_disease": ["dai thao duong", "than dai thao duong", "diabetic kidney disease"],
    "viem_cau_than_man": ["viem cau than man", "viem cau than mang", "mang tang sinh", "xo hoa cau than"],
    "benh_ly_cau_than": ["benh ly cau than", "benh cau than", "viem cau than", "cau than"],
    "benh_than_man": ["benh than man", "suy than man", "ckd", "chronic kidney disease", "kdoqi", "albumin nieu", "acr"],
}

SECTION_HINTS = {
    "definition": ["la gi", "khai niem", "dinh nghia", "co nghia la gi"],
    "classification": [
        "phan loai",
        "phan do",
        "giai doan",
        "stage",
        "class",
        "kdigo",
        "rifle",
        "failure",
        "do iv",
        "uiv",
        "sieu am",
        "a1",
        "a2",
        "a3",
        "g1",
        "g2",
        "g3",
        "g4",
        "g5",
    ],
    "diagnosis_criteria": ["chan doan", "chan doan xac dinh", "tieu chuan chan doan", "theo tieu chuan"],
    "treatment": ["dieu tri", "thuoc", "phac do", "corticosteroid", "khang sinh"],
    "clinical_features": ["trieu chung", "dau hieu", "lam sang", "can lam sang", "bieu hien", "khoi phat"],
    "pathology": ["mo benh hoc", "sinh thiet", "mien dich huynh quang", "hien vi", "lang dong"],
    "progression": ["tien trien", "tien luong", "lau dai", "tai phat"],
    "follow_up": ["theo doi", "tai kham", "du phong", "phong ngua"],
    "complications": ["bien chung", "tac dung khong mong muon", "tac dung phu"],
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
    "benh_ly_cau_than": "Bệnh lý cầu thận",
    "viem_cau_than_cap": "Viêm cầu thận cấp",
    "viem_cau_than_man": "Viêm cầu thận mạn",
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
    "complications": "Biến chứng",
}

RETRIEVAL_STOPWORDS = {
    "benh",
    "than",
    "hoi",
    "chung",
    "viem",
    "cau",
    "co",
    "la",
    "gi",
    "nao",
    "nhung",
    "cac",
    "cua",
    "cho",
    "theo",
    "duoc",
    "khi",
    "nhu",
    "the",
    "nao",
    "trong",
    "va",
    "hoac",
    "mot",
    "doi",
    "voi",
    "nguoi",
    "benh nhan",
}


class NeonVectorSearcher:
    """Thực hiện vector search trên bảng `medical_documents`."""

    def __init__(
        self,
        database_url: str,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
        bm25_retriever: BM25Retriever | None = None,
        reranker: LocalReranker | NullReranker | None = None,
    ) -> None:
        self.database_url = database_url
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.openai = AsyncOpenAI(api_key=openai_api_key)
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker or build_reranker(enabled=False)

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
        understanding["original_query"] = query
        query_embedding = await self._embed_query(understanding["embedding_query"])
        rows = await self._search_with_retry(
            embedding=query_embedding,
            top_k=top_k,
            disease_name=disease_name,
            section_type=section_type,
            source_type=source_type,
            biomarker=biomarker,
            understanding=understanding,
        )

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

    async def _search_with_retry(
        self,
        embedding: list[float],
        top_k: int,
        disease_name: str | None,
        section_type: str | None,
        source_type: str | None,
        biomarker: str | None,
        understanding: dict[str, Any],
        max_attempts: int = 2,
    ) -> list[dict[str, Any]]:
        """Retry nhẹ cho các lỗi DB connection transient khi query Neon."""

        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            conn_vector = None
            conn_keyword = None
            try:
                # Khởi tạo 2 connection riêng biệt để chạy song song thực sự mà không bị xung đột lệnh
                conn_vector = await asyncpg.connect(self.database_url, statement_cache_size=0)
                conn_keyword = await asyncpg.connect(self.database_url, statement_cache_size=0)
                
                candidate_limit = max(top_k * 6, 12)
                vector_rows = await self._search_vector_rows(
                    conn=conn_vector,
                    embedding=embedding,
                    top_k=candidate_limit,
                    disease_name=disease_name,
                    section_type=section_type,
                    source_type=source_type,
                    biomarker=biomarker,
                )
                keyword_rows = await self._search_keyword_rows(
                    conn=conn_keyword,
                    keyword_query=understanding["keyword_query"],
                    top_k=candidate_limit,
                    disease_name=disease_name,
                    section_type=section_type,
                    source_type=source_type,
                    biomarker=biomarker,
                )
                lexical_rows = await self._search_lexical_rows(
                    conn=conn_keyword,
                    understanding=understanding,
                    top_k=candidate_limit,
                    disease_name=disease_name,
                    section_type=section_type,
                    source_type=source_type,
                    biomarker=biomarker,
                )
                fused_rows = self._fuse_rows(
                    vector_rows=vector_rows,
                    keyword_rows=keyword_rows,
                    lexical_rows=lexical_rows,
                    top_k=top_k,
                    understanding=understanding,
                )
                
                # Mở rộng ngữ cảnh xung quanh các chunk (parent context expansion)
                # Dùng conn_vector tuần tự sau khi tasks song song đã hoàn thành
                expanded_rows = await self._expand_result_contexts(conn_vector, fused_rows)
                
                # Rerank dùng Cross-Encoder model (nếu được kích hoạt và không phải NullReranker)
                if self.reranker and not isinstance(self.reranker, NullReranker):
                    reranked = self.reranker.rerank(
                        query=understanding["original_query"],
                        candidates=expanded_rows,
                        top_n=top_k
                    )
                    if not reranked:
                        reranked = expanded_rows[:top_k]
                    return reranked
                else:
                    # Heuristic rerank cũ dựa trên lexical bonus
                    heuristic_reranked = self._rerank_expanded_rows(expanded_rows, understanding)
                    return heuristic_reranked[:top_k]
            except (
                asyncpg.ConnectionDoesNotExistError,
                asyncpg.ConnectionFailureError,
                asyncpg.InterfaceError,
            ) as exc:
                last_error = exc
                if attempt >= max_attempts:
                    break
                await asyncio.sleep(0.75 * attempt)
            finally:
                if conn_vector:
                    await conn_vector.close()
                if conn_keyword:
                    await conn_keyword.close()

        if last_error is not None:
            raise last_error
        raise RuntimeError("Hybrid retrieval thất bại nhưng không thu được exception cụ thể.")

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

    async def _search_lexical_rows(
        self,
        conn: asyncpg.Connection,
        understanding: dict[str, Any],
        top_k: int,
        disease_name: str | None,
        section_type: str | None,
        source_type: str | None,
        biomarker: str | None,
    ) -> list[dict[str, Any]]:
        """Vietnamese-aware substring retrieval for exact disease/criteria phrases.

        Postgres `simple` FTS is not ideal for Vietnamese medical text because it
        has no language-specific tokenizer and does not normalize accents. This
        lexical pass catches high-signal phrases such as "hội chứng thận hư",
        "thay đổi tối thiểu", "ARA 1997", or biomarker aliases.
        """

        patterns = self._lexical_patterns(understanding)
        if not patterns:
            return []

        params: list[Any] = []
        match_exprs: list[str] = []
        score_exprs: list[str] = []
        for pattern in patterns[:12]:
            params.append(f"%{pattern}%")
            index = len(params)
            match_expr = (
                f"(content ILIKE ${index} "
                f"OR source_id ILIKE ${index} "
                f"OR metadata::text ILIKE ${index})"
            )
            match_exprs.append(match_expr)
            score_exprs.append(f"CASE WHEN {match_expr} THEN 1 ELSE 0 END")

        filters: list[str] = ["(" + " OR ".join(match_exprs) + ")"]
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
        score_sql = " + ".join(score_exprs)

        sql = f"""
        SELECT
            document_id,
            source_type,
            source_id,
            content,
            metadata,
            NULL::float8 AS similarity,
            ({score_sql})::float8 AS keyword_score
        FROM medical_documents
        {where_clause}
        ORDER BY keyword_score DESC, document_id ASC
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

        safe_keyword_query = self._sanitize_keyword_query(keyword_query)
        if not safe_keyword_query:
            return []

        filters: list[str] = ["websearch_to_tsquery('simple', $1) @@ to_tsvector('simple', content)"]
        params: list[Any] = [safe_keyword_query]
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

        try:
            rows = await conn.fetch(sql, *params)
        except asyncpg.PostgresError as exc:
            # Fallback an toàn: FTS không ổn định với query quá dài/phức tạp.
            # Khi đó bỏ keyword search, vẫn giữ vector search để không làm hỏng toàn bộ /chat/answer.
            message = str(exc).lower()
            if "tsquery stack too small" in message:
                return []
            raise
        return [self._normalize_result_row(row) for row in rows]

    def _sanitize_keyword_query(self, query: str) -> str:
        """Giảm độ phức tạp tsquery để tránh lỗi `tsquery stack too small`."""
        cleaned = " ".join(str(query or "").split())
        if not cleaned:
            return ""

        # Giữ ký tự chữ/số và một số ngăn cách phổ biến để giảm toán tử tsquery.
        cleaned = re.sub(r"[^\w\s/%\-\.,:]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return ""

        # Giới hạn token + độ dài để tránh query quá sâu khi chuyển sang tsquery.
        tokens = cleaned.split(" ")
        cleaned = " ".join(tokens[:40])
        return cleaned[:300].strip()

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
            "source_file": metadata.get("source_file"),
            "chunk_index": metadata.get("chunk_index"),
            "disease_name": metadata.get("disease_name"),
            "section_type": metadata.get("section_type"),
            "doc_type": metadata.get("doc_type"),
            "biomarker": metadata.get("biomarker"),
            "preview": content[:800],
        }

    def _fuse_rows(
        self,
        vector_rows: list[dict[str, Any]],
        keyword_rows: list[dict[str, Any]],
        lexical_rows: list[dict[str, Any]],
        top_k: int,
        understanding: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Hợp nhất candidate từ vector search, FTS DB, và BM25 local theo RRF."""

        fused: dict[str, dict[str, Any]] = {}

        # 1. Fuse vector search candidates
        for rank, row in enumerate(vector_rows, start=1):
            item = fused.setdefault(row["document_id"], dict(row))
            item["vector_rank"] = rank
            item["rrf_score"] = item.get("rrf_score", 0.0) + (VECTOR_RRF_WEIGHT / (RRF_K + rank))

        # 2. Fuse FTS DB candidates
        for rank, row in enumerate(keyword_rows, start=1):
            item = fused.setdefault(row["document_id"], dict(row))
            item["keyword_rank"] = rank
            item["rrf_score"] = item.get("rrf_score", 0.0) + (KEYWORD_RRF_WEIGHT / (RRF_K + rank))
            if item.get("similarity") is None:
                item["similarity"] = row.get("similarity")
            if item.get("keyword_score") is None:
                item["keyword_score"] = row.get("keyword_score")
            if item.get("preview") is None:
                item["preview"] = row.get("preview")
            if item.get("content") is None:
                item["content"] = row.get("content")

        for rank, row in enumerate(lexical_rows, start=1):
            item = fused.setdefault(row["document_id"], dict(row))
            item["lexical_rank"] = rank
            item["rrf_score"] = item.get("rrf_score", 0.0) + (LEXICAL_RRF_WEIGHT / (RRF_K + rank))
            if item.get("similarity") is None:
                item["similarity"] = row.get("similarity")
            item["keyword_score"] = max(item.get("keyword_score") or 0.0, row.get("keyword_score") or 0.0)
            if item.get("preview") is None:
                item["preview"] = row.get("preview")
            if item.get("content") is None:
                item["content"] = row.get("content")

        for item in fused.values():
            item["metadata_bonus"] = self._metadata_bonus(item, understanding)
            item["lexical_bonus"] = self._lexical_bonus(item, understanding)
            item["fusion_score"] = round(
                item.get("rrf_score", 0.0) + item["metadata_bonus"] + item["lexical_bonus"],
                6,
            )

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
                    "bm25_score": item.get("bm25_score"),
                    "fusion_score": item["fusion_score"],
                    "metadata_bonus": item.get("metadata_bonus"),
                    "lexical_bonus": item.get("lexical_bonus"),
                    "page": item.get("page"),
                    "source_file": item.get("source_file"),
                    "chunk_index": item.get("chunk_index"),
                    "disease_name": item.get("disease_name"),
                    "section_type": item.get("section_type"),
                    "doc_type": item.get("doc_type"),
                    "biomarker": item.get("biomarker"),
                    "vector_rank": item.get("vector_rank"),
                    "keyword_rank": item.get("keyword_rank"),
                    "lexical_rank": item.get("lexical_rank"),
                    "rrf_score": item.get("rrf_score", 0.0),
                    "content": item.get("content", ""),
                    "preview": item.get("preview", ""),
                }
            )
        return results

    def _rerank_expanded_rows(
        self,
        rows: list[dict[str, Any]],
        understanding: dict[str, Any],
    ) -> list[dict[str, Any]]:
        for row in rows:
            expanded_lexical_bonus = self._lexical_bonus(row, understanding)
            row["expanded_lexical_bonus"] = expanded_lexical_bonus
            if expanded_lexical_bonus > (row.get("lexical_bonus") or 0.0):
                row["fusion_score"] = round(
                    (row.get("fusion_score") or 0.0)
                    + expanded_lexical_bonus
                    - (row.get("lexical_bonus") or 0.0),
                    6,
                )
                row["lexical_bonus"] = expanded_lexical_bonus
        return sorted(
            rows,
            key=lambda item: (
                item.get("fusion_score") or 0.0,
                item.get("keyword_score") or 0.0,
                item.get("similarity") or 0.0,
            ),
            reverse=True,
        )

    async def _expand_result_contexts(
        self,
        conn: asyncpg.Connection,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach short parent/neighbor context for chunk results.

        Many PDF chunks contain the answer but not the parent heading. Expanding
        with a nearby heading or detail chunk improves grounding without changing
        ranking.
        """

        expanded_rows: list[dict[str, Any]] = []
        for row in rows:
            if row.get("source_type") != "chunk":
                row["content"] = row.get("content") or row.get("preview", "")
                row["preview"] = (row.get("content") or row.get("preview") or "")[:800]
                expanded_rows.append(row)
                continue

            neighbor_rows = await self._fetch_neighbor_rows(conn, row)
            if neighbor_rows:
                expanded_content = self._merge_neighbor_context(row, neighbor_rows)
                if expanded_content:
                    row["original_content"] = row.get("content", "")
                    row["content"] = expanded_content
                    row["preview"] = expanded_content[:800]
                    row["expanded_context"] = True
            else:
                row["content"] = row.get("content") or row.get("preview", "")
                row["preview"] = (row.get("content") or row.get("preview") or "")[:800]
                row["expanded_context"] = False
            expanded_rows.append(row)
        return expanded_rows

    async def _fetch_neighbor_rows(
        self,
        conn: asyncpg.Connection,
        row: dict[str, Any],
    ) -> list[dict[str, Any]]:
        source_file = row.get("source_file")
        page = row.get("page")
        chunk_index = row.get("chunk_index")
        if not source_file or page is None or chunk_index is None:
            return []

        try:
            page_number = int(page)
            chunk_number = int(chunk_index)
        except (TypeError, ValueError):
            return []

        sql = """
        SELECT
            document_id,
            source_type,
            source_id,
            content,
            metadata,
            NULL::float8 AS similarity,
            NULL::float8 AS keyword_score
        FROM medical_documents
        WHERE source_type = 'chunk'
          AND metadata->>'source_file' = $1
          AND (metadata->>'page')::int = $2
          AND (metadata->>'chunk_index')::int BETWEEN $3 AND $4
        ORDER BY (metadata->>'chunk_index')::int ASC
        """
        rows = await conn.fetch(
            sql,
            source_file,
            page_number,
            max(0, chunk_number - NEIGHBOR_BACKWARD_WINDOW),
            chunk_number + NEIGHBOR_FORWARD_WINDOW,
        )
        return [self._normalize_result_row(item) for item in rows]

    def _merge_neighbor_context(
        self,
        row: dict[str, Any],
        neighbor_rows: list[dict[str, Any]],
    ) -> str:
        current_document_id = row.get("document_id")
        current_content = str(row.get("content") or row.get("preview") or "").strip()
        if not current_content:
            return ""

        selected: list[str] = []
        current_chunk_index = self._safe_int(row.get("chunk_index"))
        for neighbor in neighbor_rows:
            content = str(neighbor.get("content") or "").strip()
            if not content:
                continue
            if neighbor.get("document_id") == current_document_id:
                selected.append(content)
                continue
            neighbor_chunk_index = self._safe_int(neighbor.get("chunk_index"))
            is_previous = (
                current_chunk_index is not None
                and neighbor_chunk_index is not None
                and neighbor_chunk_index < current_chunk_index
            )
            if self._should_include_neighbor(current_content, content, neighbor, is_previous=is_previous):
                selected.append(content)

        if current_content not in selected:
            selected.append(current_content)

        merged = self._join_unique_contexts(selected)
        return merged[:MAX_RESULT_CONTEXT_CHARS].strip()

    def _should_include_neighbor(
        self,
        current_content: str,
        neighbor_content: str,
        neighbor: dict[str, Any],
        *,
        is_previous: bool,
    ) -> bool:
        """Include nearby heading/detail chunks when they add parent context."""

        text = " ".join(neighbor_content.split())
        if not text:
            return False
        neighbor_is_heading = self._looks_like_heading(text)
        current_is_heading = self._looks_like_heading(current_content)
        if is_previous and neighbor_is_heading:
            return True
        if not is_previous and (current_is_heading or len(current_content) <= 320):
            return True
        return False

    def _looks_like_heading(self, text: str) -> bool:
        cleaned = " ".join(str(text or "").split())
        if not cleaned or len(cleaned) > 220:
            return False
        return re.match(r"^\d+(?:\.\d+){0,4}\.?\s+[^\n]{3,180}$", cleaned) is not None

    def _safe_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _join_unique_contexts(self, contexts: list[str]) -> str:
        result: list[str] = []
        seen: set[str] = set()
        for context in contexts:
            cleaned = " ".join(str(context or "").split())
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
        return "\n".join(result)

    def _metadata_bonus(self, row: dict[str, Any], understanding: dict[str, Any]) -> float:
        """Cộng điểm mềm cho document khớp disease/section/biomarker hint."""

        bonus = 0.0

        if understanding["disease_hint"] and row.get("disease_name") == understanding["disease_hint"]:
            bonus += 0.08
        if understanding["section_hint"] and row.get("section_type") == understanding["section_hint"]:
            bonus += 0.10
        if understanding["section_hint"] == "definition" and row.get("section_type") == "general":
            bonus += 0.03
        if understanding["section_hint"] == "definition" and row.get("source_type") == "chunk":
            bonus += 0.01
        if understanding["biomarker_hint"] and row.get("biomarker") == understanding["biomarker_hint"]:
            bonus += 0.05

        return round(bonus, 6)

    def _lexical_bonus(self, row: dict[str, Any], understanding: dict[str, Any]) -> float:
        """Rerank nhẹ theo trùng keyword/số liệu/acronym để giảm nhiễu vector."""

        content = " ".join(
            str(value or "")
            for value in (
                row.get("content"),
                row.get("preview"),
                row.get("document_id"),
                row.get("source_id"),
                row.get("disease_name"),
                row.get("section_type"),
                row.get("biomarker"),
            )
        )
        normalized_content = self._normalize_ascii(content)
        content_tokens = set(self._content_tokens(normalized_content))

        query_tokens = understanding.get("query_tokens") or []
        overlap = len(set(query_tokens) & content_tokens)
        bonus = min(0.10, (overlap / max(1, min(len(query_tokens), 12))) * 0.10)

        number_hits = sum(1 for value in understanding.get("query_numbers") or [] if value in normalized_content)
        bonus += min(0.06, number_hits * 0.02)

        acronym_hits = sum(
            1
            for value in understanding.get("query_acronyms") or []
            if self._contains_keyword(normalized_content, value)
        )
        bonus += min(0.08, acronym_hits * 0.03)

        phrase_hits = sum(1 for phrase in understanding.get("key_phrases") or [] if phrase in normalized_content)
        bonus += min(0.08, phrase_hits * 0.04)

        return round(min(bonus, 0.22), 6)

    def _understand_query(
        self,
        query: str,
        disease_name: str | None,
        section_type: str | None,
        biomarker: str | None,
    ) -> dict[str, Any]:
        """Suy ra intent retrieval cơ bản từ query để enrich search."""

        cleaned_query = self._strip_leading_greeting(query)
        primary_query = self._primary_query_line(cleaned_query)
        normalized = self._normalize_ascii(cleaned_query)
        normalized_primary = self._normalize_ascii(primary_query)
        disease_hint = disease_name or self._detect_hint(normalized, DISEASE_HINTS)
        section_hint = section_type or self._detect_hint(normalized, SECTION_HINTS)
        biomarker_hint = biomarker or self._detect_hint(normalized, BIOMARKER_HINTS)
        query_tokens = self._content_tokens(normalized_primary)
        query_numbers = re.findall(r"\d+(?:[\.,]\d+)?", normalized_primary)
        query_acronyms = [
            token
            for token in query_tokens
            if token in {"acr", "pcr", "gfr", "egfr", "aki", "ckd", "kdigo", "kdoqi", "rifle", "ara", "uiv", "fena", "aslo"}
        ]
        key_phrases = [
            phrase
            for phrase in (
                "ara 1997",
                "kdigo 2012",
                "do iv",
                "do 4",
                "class iv",
                "failure",
                "than u nuoc",
                "viem bang quang",
                "sau nhiem lien cau",
                "thay doi toi thieu",
            )
            if phrase in normalized_primary
        ]
        lexical_patterns = self._build_lexical_patterns(
            primary_query=primary_query,
            query_tokens=query_tokens,
            key_phrases=key_phrases,
            disease_hint=disease_hint,
            section_hint=section_hint,
            biomarker_hint=biomarker_hint,
        )

        if section_hint is None and any(pattern in normalized_primary for pattern in ("la gi", "khai niem", "dinh nghia")):
            section_hint = "definition"

        embedding_lines = [f"Câu hỏi người dùng: {cleaned_query.strip()}"]
        if disease_hint:
            embedding_lines.append(f"Bệnh trọng tâm: {DISEASE_LABELS.get(disease_hint, disease_hint)}")
        if section_hint:
            embedding_lines.append(f"Mục cần tìm: {SECTION_LABELS.get(section_hint, section_hint)}")
        if biomarker_hint:
            embedding_lines.append(f"Chỉ số trọng tâm: {biomarker_hint}")

        keyword_query = primary_query.strip()
        if section_hint == "definition" and "khai niem" not in normalized_primary and "dinh nghia" not in normalized_primary:
            keyword_query = f"{keyword_query} khái niệm định nghĩa"

        return {
            "disease_hint": disease_hint,
            "section_hint": section_hint,
            "biomarker_hint": biomarker_hint,
            "embedding_query": "\n".join(embedding_lines),
            "keyword_query": keyword_query,
            "query_tokens": query_tokens,
            "query_numbers": query_numbers,
            "query_acronyms": query_acronyms,
            "key_phrases": key_phrases,
            "lexical_patterns": lexical_patterns,
        }

    def _strip_leading_greeting(self, query: str) -> str:
        """Bỏ lời chào đầu câu để retrieval tập trung vào phần y khoa."""

        cleaned = query.strip()
        cleaned = re.sub(
            r"(?iu)^\s*(?:hi|hello|hey|xin\s+chào|chào\s+bạn|chào)\s*[,!\.\-:;]*\s+",
            "",
            cleaned,
        )
        return cleaned or query.strip()

    def _primary_query_line(self, query: str) -> str:
        """Return the original user-question line from an enriched retrieval query."""

        for line in str(query or "").splitlines():
            cleaned = line.strip()
            if cleaned:
                return cleaned
        return query.strip()

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

    def _content_tokens(self, normalized_text: str) -> list[str]:
        """Token quan trọng dùng cho lexical rerank, bỏ từ quá chung."""

        tokens = re.findall(r"[a-z0-9]+", normalized_text)
        return [
            token
            for token in tokens
            if len(token) >= 3 and token not in RETRIEVAL_STOPWORDS
        ][:32]

    def _lexical_patterns(self, understanding: dict[str, Any]) -> list[str]:
        return [
            pattern
            for pattern in understanding.get("lexical_patterns", [])
            if isinstance(pattern, str) and pattern.strip()
        ]

    def _build_lexical_patterns(
        self,
        *,
        primary_query: str,
        query_tokens: list[str],
        key_phrases: list[str],
        disease_hint: str | None,
        section_hint: str | None,
        biomarker_hint: str | None,
    ) -> list[str]:
        patterns: list[str] = []
        cleaned_primary = " ".join(primary_query.split())
        if cleaned_primary and len(cleaned_primary) <= 140:
            patterns.append(cleaned_primary)

        patterns.extend(key_phrases)
        if disease_hint:
            patterns.extend(DISEASE_HINTS.get(disease_hint, []))
            label = DISEASE_LABELS.get(disease_hint)
            if label:
                patterns.append(label)
        if section_hint:
            patterns.extend(SECTION_HINTS.get(section_hint, []))
            label = SECTION_LABELS.get(section_hint)
            if label:
                patterns.append(label)
        if biomarker_hint:
            patterns.extend(BIOMARKER_HINTS.get(biomarker_hint, []))
            patterns.append(biomarker_hint)

        # Add compact token windows for user questions without exact aliases.
        important_tokens = [token for token in query_tokens if len(token) >= 4][:8]
        for index in range(0, max(0, len(important_tokens) - 1)):
            patterns.append(" ".join(important_tokens[index : index + 2]))
        for token in important_tokens[:6]:
            if token in {"kdigo", "kdoqi", "rifle", "fena", "egfr", "aslo"}:
                patterns.append(token)

        return self._unique_patterns(patterns)

    def _unique_patterns(self, patterns: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            cleaned = " ".join(str(pattern or "").strip().split())
            if not cleaned or len(cleaned) < 3:
                continue
            key = self._normalize_ascii(cleaned)
            if key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
        return result[:16]

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

    # 1. Khởi tạo BM25Retriever an toàn
    bm25_index_path = os.getenv("BM25_INDEX_PATH") or str(default_bm25_index_path())
    bm25_retriever = None
    try:
        bm25_retriever = BM25Retriever.load_if_exists(bm25_index_path)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Không thể khởi tạo BM25Retriever: {e}")

    # 2. Khởi tạo Reranker an toàn
    reranker_enabled_str = os.getenv("RERANKER_ENABLED", "true").lower()
    reranker_enabled = reranker_enabled_str in ("true", "1", "yes")
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_max_candidates = int(os.getenv("RERANKER_MAX_CANDIDATES", "12"))
    
    reranker = None
    try:
        reranker = build_reranker(
            enabled=reranker_enabled,
            model_name=reranker_model,
            max_candidates=reranker_max_candidates
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Không thể khởi tạo Reranker: {e}")

    return NeonVectorSearcher(
        database_url=database_url,
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
    )
