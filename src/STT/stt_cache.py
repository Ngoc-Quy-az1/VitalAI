"""Giữ một instance STTHandler (PhoWhisper) trong RAM để không tải lại model mỗi lần nhận dạng."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CACHE_KEY: Optional[str] = None
_HANDLER: Optional[Any] = None  # STTHandler

_KEY_FIELDS = (
    "model_name",
    "hf_model",
    "device",
    "language",
    "task",
)


def _cache_key(setup_kw: dict[str, Any]) -> str:
    sub = {k: setup_kw.get(k) for k in _KEY_FIELDS}
    return repr(sorted(sub.items()))


def get_or_create_stt_handler(**setup_kw: Any) -> Any:
    """
    Trả về ``STTHandler`` đã ``setup`` — tái sử dụng nếu cấu hình trùng lần trước.
    """
    from .STT_handler import STTHandler

    global _CACHE_KEY, _HANDLER
    key = _cache_key(setup_kw)
    if _CACHE_KEY == key and _HANDLER is not None:
        return _HANDLER

    h = STTHandler()
    h.setup(**setup_kw)
    _CACHE_KEY = key
    _HANDLER = h
    logger.info(
        "Đã tải PhoWhisper vào RAM (lần sau cùng cấu hình sẽ không load lại)."
    )
    return _HANDLER


def clear_stt_model_cache() -> None:
    global _CACHE_KEY, _HANDLER
    _CACHE_KEY = None
    _HANDLER = None
