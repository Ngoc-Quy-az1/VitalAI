from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import torch
import torchaudio

from ..VAD.base_handler import BaseHandler

logger = logging.getLogger(__name__)

try:
    import whisper

    HAS_OPENAI_WHISPER = True
except ImportError:
    HAS_OPENAI_WHISPER = False
    whisper = None  # type: ignore

try:
    from faster_whisper import WhisperModel

    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False
    WhisperModel = None  # type: ignore

Engine = Literal["faster", "openai"]


class STTHandler(BaseHandler):
    """
    STT: ưu tiên ``faster-whisper`` (nhanh, ổn định trên CPU int8), fallback ``openai-whisper``.
    """

    def setup(
        self,
        model_name: str = "small",
        engine: Engine = "faster",
        device: Optional[str] = None,
        language: str = "vi",
        task: str = "transcribe",
        fp16: bool = True,
        compute_type: Optional[str] = None,
        beam_size: int = 5,
        **kwargs: Any,
    ) -> None:
        self.language = language
        self.task = task
        self.fp16 = fp16
        self.beam_size = max(1, beam_size)
        self._fw_model = None
        self.model = None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if engine == "faster" and not HAS_FASTER_WHISPER:
            if HAS_OPENAI_WHISPER:
                logger.warning(
                    "faster-whisper not installed; falling back to openai-whisper. "
                    "Install: pip install faster-whisper"
                )
                engine = "openai"
            else:
                raise RuntimeError(
                    "Install faster-whisper (pip install faster-whisper) "
                    "or openai-whisper (pip install openai-whisper)"
                )

        if engine == "faster":
            if compute_type is None:
                compute_type = "float16" if device == "cuda" else "int8"
            logger.info(
                "Loading faster-whisper model=%s device=%s compute_type=%s",
                model_name,
                device,
                compute_type,
            )
            self._fw_model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
            self._engine = "faster"
        else:
            if not HAS_OPENAI_WHISPER:
                raise RuntimeError(
                    "openai-whisper not installed. Run: pip install openai-whisper"
                )
            logger.info("Loading openai-whisper model=%s device=%s", model_name, device)
            self.model = whisper.load_model(model_name, device=device)
            self._engine = "openai"

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

        if self._engine == "faster" and self._fw_model is not None:
            segments, _ = self._fw_model.transcribe(
                x.astype(np.float32),
                language=self.language if self.language else None,
                task=self.task,
                beam_size=self.beam_size,
                vad_filter=False,
            )
            parts = [s.text for s in segments]
            return "".join(parts).strip()

        if self.model is None:
            raise RuntimeError("STT model not loaded.")

        result = self.model.transcribe(
            x,
            language=self.language if self.language else None,
            task=self.task,
            fp16=self.fp16 and self.device == "cuda",
            verbose=False,
        )
        return (result.get("text") or "").strip()

    def process(self, audio_chunk):
        raise NotImplementedError(
            "Use transcribe() on a full audio segment after VAD."
        )

    def on_session_end(self):
        pass 


ASRHandler = STTHandler