from __future__ import annotations

"""
Bộ indexer tối thiểu để đưa embedding lên Neon + pgvector.

Trách nhiệm của file này:
- đọc `embedding_documents.jsonl`
- đảm bảo bảng/vector index tồn tại
- gọi OpenAI Embeddings theo batch
- upsert document vào `medical_documents`

File này chưa làm:
- hybrid retrieval
- reranking
- queue/job orchestration
- cost tracking chi tiết
"""

import json
import os
from pathlib import Path
from typing import Any

import asyncpg
from dotenv import load_dotenv
from openai import AsyncOpenAI


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Đọc JSONL UTF-8 thành list dict."""

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class NeonEmbeddingIndexer:
    """Index embedding documents lên Neon bằng pgvector."""

    def __init__(
        self,
        input_path: str | Path,
        database_url: str,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
        batch_size: int = 50,
    ) -> None:
        self.input_path = Path(input_path)
        self.database_url = database_url
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.batch_size = batch_size
        self.openai = AsyncOpenAI(api_key=openai_api_key)

    async def run(self, limit: int | None = None) -> dict[str, Any]:
        """Chạy toàn bộ pipeline ensure schema -> embed -> upsert."""

        documents = load_jsonl(self.input_path)
        if limit is not None:
            documents = documents[:limit]

        conn = await asyncpg.connect(self.database_url, statement_cache_size=0)
        try:
            await self._ensure_schema(conn)
            inserted = 0
            for start in range(0, len(documents), self.batch_size):
                batch = documents[start : start + self.batch_size]
                embeddings = await self._embed_batch(batch)
                await self._upsert_batch(conn, batch, embeddings)
                inserted += len(batch)

            total_rows = await conn.fetchval("SELECT COUNT(*) FROM medical_documents")
            return {
                "input_path": str(self.input_path),
                "embedding_model": self.embedding_model,
                "embedding_dimensions": self.embedding_dimensions,
                "batch_size": self.batch_size,
                "processed_documents": len(documents),
                "upserted_documents": inserted,
                "rows_in_table": total_rows,
            }
        finally:
            await conn.close()

    async def _embed_batch(self, batch: list[dict[str, Any]]) -> list[list[float]]:
        """Gọi OpenAI Embeddings cho một batch document."""

        inputs = [item["embedding_text"] for item in batch]
        response = await self.openai.embeddings.create(
            model=self.embedding_model,
            input=inputs,
            dimensions=self.embedding_dimensions,
        )
        return [item.embedding for item in response.data]

    async def _ensure_schema(self, conn: asyncpg.Connection) -> None:
        """Tạo extension, bảng và index cần thiết nếu chưa tồn tại."""

        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            "CREATE EXTENSION IF NOT EXISTS pgcrypto",
            "CREATE EXTENSION IF NOT EXISTS pg_trgm",
            f"""
            CREATE TABLE IF NOT EXISTS medical_documents (
                id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id     TEXT UNIQUE NOT NULL,
                source_type     TEXT NOT NULL,
                source_id       TEXT NOT NULL,
                content         TEXT NOT NULL,
                embedding       VECTOR({self.embedding_dimensions}),
                metadata        JSONB NOT NULL DEFAULT '{{}}',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS medical_documents_metadata_gin_idx
            ON medical_documents
            USING gin (metadata)
            """,
            """
            CREATE INDEX IF NOT EXISTS medical_documents_content_fts_idx
            ON medical_documents
            USING gin (to_tsvector('simple', content))
            """,
            """
            CREATE INDEX IF NOT EXISTS medical_documents_embedding_hnsw_idx
            ON medical_documents
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """,
        ]
        for statement in statements:
            await conn.execute(statement)

    async def _upsert_batch(
        self,
        conn: asyncpg.Connection,
        batch: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert một batch document + vector vào bảng đích."""

        sql = """
        INSERT INTO medical_documents (
            document_id,
            source_type,
            source_id,
            content,
            embedding,
            metadata
        )
        VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb)
        ON CONFLICT (document_id) DO UPDATE SET
            source_type = EXCLUDED.source_type,
            source_id = EXCLUDED.source_id,
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata
        """

        async with conn.transaction():
            for item, embedding in zip(batch, embeddings):
                await conn.execute(
                    sql,
                    item["document_id"],
                    item["source_type"],
                    item["source_id"],
                    item["content"],
                    json.dumps(embedding),
                    json.dumps(item["metadata"], ensure_ascii=False),
                )


def build_indexer_from_env(
    input_path: str | Path,
    batch_size: int = 50,
) -> NeonEmbeddingIndexer:
    """Khởi tạo indexer từ `.env` hoặc env hiện tại."""

    load_dotenv()

    database_url = os.getenv("NEON_DATABASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

    if not database_url:
        raise ValueError("Thiếu biến môi trường NEON_DATABASE_URL")
    if not openai_api_key:
        raise ValueError("Thiếu biến môi trường OPENAI_API_KEY")

    return NeonEmbeddingIndexer(
        input_path=input_path,
        database_url=database_url,
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        batch_size=batch_size,
    )

