from __future__ import annotations

import re


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR text for downstream parsing and prompting."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\t", " ")
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("‘", "'").replace("’", "'")

    lines = []
    for line in normalized.split("\n"):
        compact = re.sub(r"\s+", " ", line).strip()
        if compact:
            lines.append(compact)
    return "\n".join(lines)
