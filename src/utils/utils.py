import numpy as np


def int2float(sound: np.ndarray) -> np.ndarray:
    """Chuyển PCM int16 sang float32 trong [-1, 1] (chuẩn cho VAD/STT)."""
    return sound.astype(np.float32) / 32768.0
