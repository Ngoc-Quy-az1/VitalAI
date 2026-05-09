from __future__ import annotations

import re
import unicodedata
from typing import Any


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def format_number(value: Any) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number.is_integer():
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def join_value_unit(name: str, value: str, unit: str | None) -> str:
    return f"{name} {value}{f' {unit}' if unit else ''}"


def normalize_for_match(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    normalized = normalized.replace("đ", "d")
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^\w\s]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()
