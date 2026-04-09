from .STT_handler import STTHandler

# Backward compatibility (same class as STTHandler)
ASRHandler = STTHandler
from .vad_STT_pipeline import (
    VADASRPipeline,
    VADSTTPipeline,
    iter_speech_segments_from_vad,
)

__all__ = [
    "STTHandler",
    "ASRHandler",
    "VADSTTPipeline",
    "VADASRPipeline",
    "iter_speech_segments_from_vad",
]
