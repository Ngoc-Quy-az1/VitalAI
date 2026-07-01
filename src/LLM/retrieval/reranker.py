from __future__ import annotations

"""
LocalReranker — Reranker Cross-Encoder chạy hoàn toàn trên CPU, 0đ chi phí.

Mô hình mặc định: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Kích thước: ~80MB
  - Tốc độ: ~50–150ms cho 12 candidates trên CPU thường
  - Chất lượng: Tốt cho tiếng Anh và các thuật ngữ y khoa quốc tế
  - Ngôn ngữ: Hỗ trợ truy vấn tiếng Việt (cross-lingual tốt với query ngắn)

Để cải thiện tiếng Việt hơn, có thể đổi sang:
  - "BAAI/bge-reranker-base" — đa ngôn ngữ, hỗ trợ Việt tốt hơn
  - "DiLiCo/vietnamese-reranker" — chuyên tiếng Việt (nếu có)

Quy trình pipeline:
    candidates_union → Reranker.rerank(query, candidates, top_n) → top_n chunks
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Số candidates tối đa đưa vào reranker để giữ latency < 150ms trên CPU
_RERANKER_MAX_CANDIDATES = 12

# Ngưỡng điểm sigmoid tối thiểu — loại bỏ kết quả hoàn toàn không liên quan
_RERANKER_MIN_SCORE = -10.0  # logit score (cross-encoder output), ~0.000045 sau sigmoid

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LocalReranker:
    """
    Cross-Encoder Reranker chạy trực tiếp trên CPU — không cần GPU, không cần API.

    Tự động tải mô hình từ Hugging Face trong lần khởi động đầu tiên (~80MB).
    Các lần sau load từ cache cục bộ (~2–3 giây).

    Sử dụng:
        reranker = LocalReranker()
        reranked = reranker.rerank(query, candidates, top_n=5)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
        max_candidates: int = _RERANKER_MAX_CANDIDATES,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_candidates = max_candidates
        self._model: Any = None  # Lazy load — chỉ tải khi lần đầu gọi rerank()

    def _ensure_model_loaded(self) -> None:
        """Lazy load mô hình Cross-Encoder lần đầu sử dụng."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers chưa được cài. "
                "Chạy: pip install sentence-transformers"
            ) from exc

        logger.info(
            "LocalReranker: Đang tải mô hình '%s' trên device='%s' ...",
            self._model_name,
            self._device,
        )
        self._model = CrossEncoder(self._model_name, device=self._device)
        logger.info("LocalReranker: Mô hình đã sẵn sàng.")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Rerank danh sách candidates theo độ liên quan với query.

        Luôn giới hạn đầu vào tối đa _max_candidates để kiểm soát latency.

        Args:
            query: Câu hỏi người dùng gốc.
            candidates: Danh sách chunk dict từ bước fusion (đã có content).
            top_n: Số kết quả trả về sau reranking.

        Returns:
            Danh sách chunk dict được sắp xếp lại, có thêm key 'rerank_score'.
        """
        if not candidates:
            return []

        self._ensure_model_loaded()

        # Hard limit candidates để giữ latency ổn định
        input_candidates = candidates[: self._max_candidates]
        if len(candidates) > self._max_candidates:
            logger.debug(
                "LocalReranker: Cắt candidates từ %d xuống %d để giữ latency.",
                len(candidates),
                self._max_candidates,
            )

        # Tạo cặp [query, content] cho cross-encoder
        pairs = [
            [query, str(c.get("content") or c.get("preview") or "")]
            for c in input_candidates
        ]

        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            logger.error(
                "LocalReranker: Lỗi trong quá trình predict: %s. Fallback về thứ tự gốc.", exc
            )
            return candidates[:top_n]

        # Gán điểm và sắp xếp
        for idx, score in enumerate(scores):
            input_candidates[idx]["rerank_score"] = round(float(score), 6)

        # Lọc bỏ kết quả không liên quan và sắp xếp giảm dần
        ranked = sorted(
            [c for c in input_candidates if c.get("rerank_score", -999) >= _RERANKER_MIN_SCORE],
            key=lambda x: x.get("rerank_score", 0.0),
            reverse=True,
        )

        logger.debug(
            "LocalReranker: Đã rerank %d candidates, trả về top %d. "
            "Top score=%.4f",
            len(input_candidates),
            min(top_n, len(ranked)),
            ranked[0].get("rerank_score", 0.0) if ranked else 0.0,
        )
        return ranked[:top_n]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


class NullReranker:
    """
    Reranker giả — trả về candidates nguyên vẹn theo thứ tự gốc.
    Dùng khi sentence-transformers chưa được cài hoặc khi disabled.
    """

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        return candidates[:top_n]

    @property
    def is_loaded(self) -> bool:
        return False


def build_reranker(
    enabled: bool = True,
    model_name: str = _DEFAULT_MODEL,
    device: str = "cpu",
    max_candidates: int = _RERANKER_MAX_CANDIDATES,
) -> LocalReranker | NullReranker:
    """
    Factory function tạo reranker phù hợp.

    Nếu enabled=False hoặc sentence-transformers chưa cài,
    trả về NullReranker để không làm crash hệ thống.
    """
    if not enabled:
        logger.info("LocalReranker: Đã tắt theo cấu hình. Dùng NullReranker.")
        return NullReranker()

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        logger.warning(
            "LocalReranker: sentence-transformers chưa cài. "
            "Dùng NullReranker. Cài: pip install sentence-transformers"
        )
        return NullReranker()

    return LocalReranker(
        model_name=model_name,
        device=device,
        max_candidates=max_candidates,
    )
