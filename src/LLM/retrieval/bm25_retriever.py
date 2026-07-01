from __future__ import annotations

"""
BM25Retriever — Tìm kiếm từ khóa cục bộ trên tập corpus y khoa.

Chạy hoàn toàn trong RAM, không cần DB hay API bên ngoài.
Tốc độ tìm kiếm: ~5–15ms cho corpus ~10.000 chunks.

Quy trình sử dụng:
1. Lúc ingestion: gọi BM25IndexBuilder.build_and_save(chunks, path)
2. Lúc khởi động API: gọi BM25Retriever.load(path)
3. Lúc retrieval: gọi retriever.search(query, top_k)
"""

import logging
import os
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BM25_MIN_SCORE = 0.01

_MAX_QUERY_TOKENS = 32

_MEDICAL_STOPWORDS = frozenset(
    [
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
        "trong",
        "va",
        "hoac",
        "mot",
        "doi",
        "voi",
        "nguoi",
        "ban",
        "toi",
        "day",
        "nay",
        "do",
        "de",
        "thi",
        "se",
        "da",
        "dang",
        "duoi",
        "tren",
        "nam",
        "vao",
        "ra",
        "len",
        "xuong",
        "sau",
        "truoc",
        "hay",
        "khong",
        "co",
    ]
)


def _normalize_vi(text: str) -> str:
    """Bỏ dấu tiếng Việt, lowercase, collapse khoảng trắng."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii").lower()
    return " ".join(ascii_text.split())


def _tokenize(text: str) -> list[str]:
    """
    Tokenize văn bản y khoa tiếng Việt đã normalize.
    Giữ lại: từ >= 2 ký tự, số thập phân, acronym y khoa.
    """
    normalized = _normalize_vi(text)
    raw_tokens = re.findall(r"[a-z0-9]+(?:[./\-][a-z0-9]+)*", normalized)
    tokens: list[str] = []
    for token in raw_tokens:
        # Giữ số thập phân như "1.73", "16000"
        if re.fullmatch(r"\d+(?:[.,]\d+)?", token):
            tokens.append(token)
            continue
        # Bỏ token ngắn hoặc stopword y khoa
        if len(token) < 2 or token in _MEDICAL_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


class BM25IndexBuilder:
    """Xây dựng và lưu chỉ mục BM25 ra file để dùng lại."""

    @staticmethod
    def build_and_save(chunks: list[dict[str, Any]], save_path: str | Path) -> None:
        """
        Xây dựng chỉ mục BM25Okapi từ danh sách chunks và lưu ra file .pkl.

        Args:
            chunks: Danh sách dict, mỗi dict phải có ít nhất key 'content'.
            save_path: Đường dẫn file .pkl để lưu.
        """
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "rank_bm25 chưa được cài. Chạy: pip install rank-bm25"
            ) from exc

        logger.info("BM25IndexBuilder: Bắt đầu tokenize %d chunks...", len(chunks))
        tokenized_corpus = []
        valid_chunks = []

        for chunk in chunks:
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            tokens = _tokenize(content)
            if not tokens:
                continue
            tokenized_corpus.append(tokens)
            valid_chunks.append(chunk)

        if not tokenized_corpus:
            logger.warning("BM25IndexBuilder: Không có chunk hợp lệ nào để index.")
            return

        logger.info(
            "BM25IndexBuilder: Đang build BM25Okapi cho %d chunks...",
            len(valid_chunks),
        )
        bm25 = BM25Okapi(tokenized_corpus)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "bm25": bm25,
                    "chunks": valid_chunks,
                    "num_chunks": len(valid_chunks),
                    "version": "1.0",
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        logger.info(
            "BM25IndexBuilder: Đã lưu index (%d chunks) -> %s",
            len(valid_chunks),
            save_path,
        )


class BM25Retriever:
    """
    Retriever tìm kiếm từ khóa BM25 chạy hoàn toàn trên RAM.

    Sử dụng:
        retriever = BM25Retriever.load("data/processed_data/bm25_index.pkl")
        results = retriever.search("GFR bao nhiêu là bệnh thận mạn", top_k=15)
    """

    def __init__(self, bm25: Any, chunks: list[dict[str, Any]]) -> None:
        self._bm25 = bm25
        self._chunks = chunks
        logger.info(
            "BM25Retriever: Đã khởi tạo với %d chunks trong RAM.", len(chunks)
        )

    @classmethod
    def load(cls, index_path: str | Path) -> "BM25Retriever":
        """Load chỉ mục BM25 đã được build trước từ file .pkl."""
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index không tìm thấy tại: {index_path}. "
                "Hãy chạy BM25IndexBuilder.build_and_save() trước khi khởi động API."
            )

        try:
            from rank_bm25 import BM25Okapi  # noqa: F401 — kiểm tra thư viện tồn tại
        except ImportError as exc:
            raise ImportError(
                "rank_bm25 chưa được cài. Chạy: pip install rank-bm25"
            ) from exc

        logger.info("BM25Retriever: Đang load index từ %s ...", index_path)
        with open(index_path, "rb") as f:
            data = pickle.load(f)

        return cls(bm25=data["bm25"], chunks=data["chunks"])

    @classmethod
    def load_if_exists(cls, index_path: str | Path) -> "BM25Retriever | None":
        """Load index nếu file tồn tại, trả None nếu không có để fallback an toàn."""
        try:
            return cls.load(index_path)
        except FileNotFoundError:
            logger.warning(
                "BM25Retriever: Không tìm thấy index tại %s. "
                "BM25 sẽ bị bỏ qua trong lần chạy này.",
                index_path,
            )
            return None

    def search(self, query: str, top_k: int = 15) -> list[dict[str, Any]]:
        """
        Tìm kiếm BM25 đồng bộ — cực nhanh (~5–15ms).

        Args:
            query: Câu hỏi người dùng (tiếng Việt có hoặc không có dấu đều được).
            top_k: Số kết quả tối đa trả về.

        Returns:
            Danh sách chunk dict có thêm key 'bm25_score'.
        """
        tokens = _tokenize(query)
        if not tokens:
            logger.debug("BM25Retriever: Query rỗng sau tokenize, bỏ qua.")
            return []

        # Giới hạn token để tránh overhead tính điểm
        tokens = tokens[:_MAX_QUERY_TOKENS]

        try:
            scores = self._bm25.get_scores(tokens)
        except Exception as exc:
            logger.error("BM25Retriever: Lỗi tính điểm BM25: %s", exc)
            return []

        # Lấy top_k index có điểm cao nhất, lọc bỏ chunk điểm quá thấp
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < _BM25_MIN_SCORE:
                break  # Đã sort giảm dần, các phần tử sau cũng thấp hơn
            chunk = dict(self._chunks[idx])
            chunk["bm25_score"] = round(score, 6)
            chunk["keyword_score"] = round(score, 6)  # Alias cho tương thích fusion
            results.append(chunk)

        logger.debug(
            "BM25Retriever: Query='%s...' -> %d kết quả (top_k=%d)",
            query[:60],
            len(results),
            top_k,
        )
        return results

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)


def default_bm25_index_path() -> Path:
    """Đường dẫn mặc định cho BM25 index trong project VitalAI."""
    base = Path(os.getenv("MEDICAL_TOOLS_DATA_DIR", "data/processed_data"))
    return base / "bm25_index.pkl"
