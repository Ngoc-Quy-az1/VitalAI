from __future__ import annotations

import re


TTS_REPLACEMENTS = {
    "KidneyCare AI": "Kidney Care ây ai",
    "AI": "ây ai",
    "CKD": "bệnh thận mạn",
    "AKI": "suy thận cấp",
    "GFR": "mức lọc cầu thận",
    "eGFR": "mức lọc cầu thận ước tính",
    "MDRD": "em đi a rờ đi",
    "KDIGO": "ca đi gô",
    "RIFLE": "rai phồ",
    "ACR": "a c rờ",
    "PCR": "p c rờ",
    "BSA": "b s a",
}


def _apply_tts_replacements(text: str) -> str:
    output = text
    for source, target in TTS_REPLACEMENTS.items():
        output = re.sub(rf"\b{re.escape(source)}\b", target, output)
    return output


def prepare_tts_text(text: str) -> str:
    """Làm sạch markdown/ký hiệu để Web Speech đọc tự nhiên hơn."""
    cleaned = text or ""
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"#+\s*", "", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
    cleaned = re.sub(r"\$\$[\s\S]*?\$\$", " công thức toán học ", cleaned)
    cleaned = re.sub(r"\$([^$]+)\$", r"\1", cleaned)
    cleaned = _apply_tts_replacements(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

