from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from mistralai.client import Mistral

from src.OCR.text_normalizer import normalize_ocr_text


class MistralOcrError(RuntimeError):
    """Raised when Mistral OCR fails or returns empty content."""


def _load_image_base64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
load_dotenv()

def _get_mistral_client() -> Mistral:
    api_key = os.getenv("MISTRAL_CLIENT_API_KEY")
    if not api_key:
        raise MistralOcrError("Thiếu API key trong môi trường.")
    return Mistral(api_key=api_key)


def run_mistral_ocr(
    *,
    image_path: Path,
    language: str = "vi",
    model: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    if not image_path.exists():
        raise MistralOcrError(f"Không tìm thấy ảnh đầu vào: {image_path}")

    image_b64 = _load_image_base64(image_path)
    model_name = model or os.getenv("MISTRAL_OCR_MODEL_NAME", "pixtral-12b-latest")

    sys_prompt = system_prompt or (
        "Bạn là bác sĩ trợ lý. Hãy đọc ảnh phiếu khám / xét nghiệm và TRẢ VỀ NGUYÊN VĂN NỘI DUNG dạng text thuần, "
        "giữ lại tất cả chỉ số, đơn vị, tên xét nghiệm, cột, hàng. Không diễn giải, không tóm tắt, không thêm lời "
        "khuyên. Chỉ in lại nội dung như đang gõ lại phiếu. Ngôn ngữ đầu ra ưu tiên tiếng Việt nếu phiếu là tiếng Việt."
    )

    mime = "image/png"
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"

    image_data_url = f"data:{mime};base64,{image_b64}"
    user_text = (
        "Hãy OCR toàn bộ nội dung trong ảnh phiếu khám / xét nghiệm này và in ra dạng text thuần, "
        "giữ nguyên số liệu, đơn vị, tiêu đề cột, dòng. Không cần giải thích thêm."
    )
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    }
    system_message = {"role": "system", "content": sys_prompt}

    try:
        client = _get_mistral_client()
        response = client.chat.complete(
            model=model_name,
            messages=[system_message, user_message],
            temperature=0.0,
            max_tokens=2048,
        )
    except Exception as exc:
        # Fall back to OpenAI GPT-4o for OCR if Mistral fails/is unauthorized
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise MistralOcrError(f"Gọi Mistral OCR thất bại: {exc} và không tìm thấy OPENAI_API_KEY để fallback.") from exc
        
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Mistral OCR failed (%s). Falling back to OpenAI GPT-4o-mini...", exc)
        
        try:
            import openai
            openai_client = openai.OpenAI(api_key=openai_key)
            
            # Format message for OpenAI vision
            openai_user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
            
            openai_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": openai_user_content}
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            raw = openai_response.choices[0].message.content or ""
            normalized = normalize_ocr_text(raw)
            if not normalized:
                raise MistralOcrError("OpenAI OCR fallback không trả về nội dung văn bản.")
            
            return {
                "text": normalized,
                "raw_text": raw,
                "model": "gpt-4o-mini",
                "language": language,
            }
        except Exception as oai_exc:
            raise MistralOcrError(f"Cả Mistral OCR và OpenAI OCR fallback đều thất bại. Mistral error: {exc}. OpenAI error: {oai_exc}") from oai_exc


    try:
        choice0 = response.choices[0]
        content = choice0.message.content if choice0.message else None  # type: ignore[union-attr]
        if isinstance(content, str):
            raw = content
        elif isinstance(content, list):
            # Content is a list of chunks (text/image/thinking...)
            text_parts: list[str] = []
            for chunk in content:
                if isinstance(chunk, str):
                    text_parts.append(chunk)
                    continue
                chunk_text = getattr(chunk, "text", None)
                if chunk_text:
                    text_parts.append(str(chunk_text))
                    continue
                # If chunk is dict-like
                if isinstance(chunk, dict) and chunk.get("type") == "text" and chunk.get("text"):
                    text_parts.append(str(chunk["text"]))
            raw = "\n".join(text_parts)
        else:
            raw = str(content or "")
    except Exception as exc:
        raise MistralOcrError(f"Response Mistral không đúng định dạng mong đợi: {exc}") from exc

    normalized = normalize_ocr_text(raw or "")
    if not normalized:
        raise MistralOcrError("Mistral OCR không trả về nội dung văn bản.")

    return {
        "text": normalized,
        "raw_text": raw,
        "model": model_name,
        "language": language,
    }

