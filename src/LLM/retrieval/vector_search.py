from __future__ import annotations

"""
Retriever tối thiểu để test semantic search trực tiếp trên Neon + pgvector.

File này dành cho giai đoạn kiểm tra chất lượng retrieval:
- embed câu hỏi người dùng
- query top-k theo cosine similarity
- hỗ trợ filter metadata cơ bản

Chưa làm ở bước này:
- hybrid search với full-text
- RRF
- reranker
- query rewriting
"""

import json
import os
from typing import Any

import asyncpg
from dotenv import load_dotenv
from openai import AsyncOpenAI


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
        """Embed query và trả về top-k kết quả tương tự nhất."""

        query_embedding = await self._embed_query(query)
        conn = await asyncpg.connect(self.database_url, statement_cache_size=0)
        try:
            rows = await self._search_rows(
                conn=conn,
                embedding=query_embedding,
                top_k=top_k,
                disease_name=disease_name,
                section_type=section_type,
                source_type=source_type,
                biomarker=biomarker,
            )
        finally:
            await conn.close()

        return {
            "query": query,
            "top_k": top_k,
            "filters": {
                "disease_name": disease_name,
                "section_type": section_type,
                "source_type": source_type,
                "biomarker": biomarker,
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

    async def _search_rows(
        self,
        conn: asyncpg.Connection,
        embedding: list[float],
        top_k: int,
        disease_name: str | None,
        section_type: str | None,
        source_type: str | None,
        biomarker: str | None,
    ) -> list[dict[str, Any]]:
        """Thực hiện câu SQL search với các filter metadata tùy chọn."""

        filters: list[str] = []
        params: list[Any] = [json.dumps(embedding)]

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
            1 - (embedding <=> $1::vector) AS similarity
        FROM medical_documents
        {where_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT ${limit_index}
        """

        rows = await conn.fetch(sql, *params)
        results: list[dict[str, Any]] = []
        for row in rows:
            metadata = row["metadata"] or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            content = row["content"]
            results.append(
                {
                    "document_id": row["document_id"],
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "similarity": round(float(row["similarity"]), 6),
                    "page": metadata.get("page"),
                    "disease_name": metadata.get("disease_name"),
                    "section_type": metadata.get("section_type"),
                    "doc_type": metadata.get("doc_type"),
                    "biomarker": metadata.get("biomarker"),
                    "preview": content[:500],
                }
            )
        return results


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
