from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import torchaudio

from ..VAD.base_handler import BaseHandler

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline as hf_asr_pipeline

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    hf_asr_pipeline = None  # type: ignore[misc, assignment]

# VinAI PhoWhisper (Hugging Face) — engine STT duy nhất trong VitalAI
DEFAULT_PHOWHISPER_MODEL = "vinai/PhoWhisper-base"


class STTHandler(BaseHandler):
    """Nhận dạng giọng nói qua PhoWhisper (transformers ASR pipeline)."""

    def setup(
        self,
        *,
        model_name: str = DEFAULT_PHOWHISPER_MODEL,
        hf_model: Optional[str] = None,
        device: Optional[str] = None,
        language: str = "vi",
        task: str = "transcribe",
        **kwargs: Any,
    ) -> None:
        self.language = language
        self.task = task
        self._hf_pipe = None

        if not HAS_TRANSFORMERS or hf_asr_pipeline is None:
            raise RuntimeError(
                "PhoWhisper cần transformers. Cài: pip install transformers torch"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model_id = hf_model or os.getenv("PHOWHISPER_MODEL") or model_name
        if not model_id or "/" not in model_id:
            model_id = DEFAULT_PHOWHISPER_MODEL

        pipe_device = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
        if device == "cuda" and pipe_device == -1:
            logger.warning("CUDA không khả dụng — PhoWhisper chạy CPU (rất chậm).")

        logger.info(
            "Loading PhoWhisper model=%s device=%s",
            model_id,
            pipe_device,
        )
        self._hf_pipe = hf_asr_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=pipe_device,
        )

    def _ensure_mono_float32(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=-1)
        return np.clip(x, -1.0, 1.0)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        x = self._ensure_mono_float32(audio)
        if sample_rate != 16000:
            w = torch.from_numpy(x).unsqueeze(0)
            x = (
                torchaudio.functional.resample(
                    w, orig_freq=sample_rate, new_freq=16000
                )
                .squeeze(0)
                .numpy()
            )

        if self._hf_pipe is None:
            raise RuntimeError("Gọi setup() trước khi transcribe.")

        wlang = self.language or None
        if wlang == "vi":
            wlang = "vietnamese"
        gen_kw: dict[str, Any] = {"task": self.task}
        if wlang:
            gen_kw["language"] = wlang
        out = self._hf_pipe(
            {"raw": x.astype(np.float32), "sampling_rate": 16000},
            generate_kwargs=gen_kw,
        )
        if isinstance(out, dict):
            return (out.get("text") or "").strip()
        return str(out).strip()

    def process(self, audio_chunk):
        raise NotImplementedError(
            "Use transcribe() on a full audio segment after VAD."
        )

    def on_session_end(self):
        pass


ASRHandler = STTHandler
