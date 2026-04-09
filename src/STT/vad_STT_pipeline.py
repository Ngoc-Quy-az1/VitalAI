
from __future__ import annotations

import logging
from threading import Event
from typing import Generator, Iterator, Optional

import numpy as np

from ..VAD.vad_handler import VADHandler
from .stt_cache import get_or_create_stt_handler

logger = logging.getLogger(__name__)


class VADSTTPipeline:
    def __init__(self):
        self.vad: Optional[VADHandler] = None
        self.STT: Optional[object] = None

    def setup(
        self,
        should_listen: Optional[Event] = None,
        vad_kwargs: Optional[dict] = None,
        STT_kwargs: Optional[dict] = None,
        stt_kwargs: Optional[dict] = None,
        asr_kwargs: Optional[dict] = None,
    ):
        if should_listen is None:
            should_listen = Event()
            should_listen.set()

        vad_kwargs = vad_kwargs or {}
        if STT_kwargs is None:
            STT_kwargs = stt_kwargs if stt_kwargs is not None else asr_kwargs
        STT_kwargs = STT_kwargs or {}

        self.vad = VADHandler()
        self.vad.setup(should_listen, **vad_kwargs)

        self.STT = get_or_create_stt_handler(**STT_kwargs)

    def process_audio_chunk(
        self, pcm_int16_bytes: bytes
    ) -> Generator[str, None, None]:
        if self.vad is None or self.STT is None:
            raise RuntimeError("Gọi setup() trước.")

        for item in self.vad.process(pcm_int16_bytes):
            if isinstance(item, tuple) and len(item) == 2:
                kind, audio = item
                if kind != "final":
                    continue
            else:
                audio = item

            text = self.STT.transcribe(audio, self.vad.sample_rate)
            if text:
                logger.info("STT: %s", text)
                yield text

    def reset_session(self):
        if self.vad:
            self.vad.on_session_end()


def iter_speech_segments_from_vad(
    vad: VADHandler, pcm_stream: Iterator[bytes]
) -> Generator[np.ndarray, None, None]:
    """Từ iterator chunk PCM, yield từng đoạn numpy sau VAD (chế độ normal)."""
    for chunk in pcm_stream:
        for seg in vad.process(chunk):
            if isinstance(seg, tuple):
                _, arr = seg
                yield arr
            else:
                yield seg


VADASRPipeline = VADSTTPipeline
