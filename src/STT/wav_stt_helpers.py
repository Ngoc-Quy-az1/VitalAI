"""Tải WAV, ghi WAV, STT / VAD+STT (PhoWhisper) — dùng chung cho CLI và GUI."""

from __future__ import annotations

import wave
from pathlib import Path
from threading import Event
from typing import Optional

import numpy as np

from .STT_handler import DEFAULT_PHOWHISPER_MODEL

SAMPLE_RATE = 16000


def _pcm_bytes_to_float_mono(raw: bytes, n_channels: int, sampwidth: int) -> np.ndarray:
    """PCM little-endian → float32 mono [-1, 1]."""
    if sampwidth == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Định dạng WAV không hỗ trợ (sampwidth={sampwidth}). Dùng PCM 8/16/32 bit.")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)
    return x


def _load_wav_file_stdlib(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    x = _pcm_bytes_to_float_mono(raw, n_channels, sampwidth)
    return x, sr


def _resample_audio(x: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    if orig_sr == new_sr:
        return x
    import torch
    import torchaudio

    w = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    y = torchaudio.functional.resample(w, orig_freq=orig_sr, new_freq=new_sr)
    return y.squeeze(0).numpy()


def load_wav_mono(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Đọc WAV → mono float32 @ target_sr.
    File .wav dùng module ``wave`` (stdlib) để tránh lỗi torchaudio thiếu backend trên Windows.
    Định dạng khác thử ``torchaudio.load``.
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        x, sr = _load_wav_file_stdlib(path)
        return _resample_audio(x, sr, target_sr)

    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy()


def float_audio_to_pcm16_bytes(audio_f32: np.ndarray) -> bytes:
    clip = np.clip(audio_f32.astype(np.float32), -1.0, 1.0)
    pcm = (clip * 32767.0).astype(np.int16)
    return pcm.tobytes()


def transcribe_wav_file(
    wav_path: Path,
    *,
    hf_model: Optional[str] = None,
    model_name: str = DEFAULT_PHOWHISPER_MODEL,
) -> str:
    from .stt_cache import get_or_create_stt_handler

    kw: dict = {
        "model_name": model_name,
        "language": "vi",
    }
    if hf_model:
        kw["hf_model"] = hf_model
    h = get_or_create_stt_handler(**kw)
    audio = load_wav_mono(wav_path)
    return h.transcribe(audio, SAMPLE_RATE)


def transcribe_wav_with_vad(
    wav_path: Path,
    *,
    hf_model: Optional[str] = None,
    model_name: str = DEFAULT_PHOWHISPER_MODEL,
) -> list[str]:
    from .vad_STT_pipeline import VADSTTPipeline

    audio = load_wav_mono(wav_path)
    pcm = float_audio_to_pcm16_bytes(audio)
    chunk_samples = 512
    chunk_bytes = chunk_samples * 2

    ev = Event()
    ev.set()
    pipe = VADSTTPipeline()
    stt_kw: dict = {
        "model_name": model_name,
        "language": "vi",
    }
    if hf_model:
        stt_kw["hf_model"] = hf_model
    pipe.setup(
        should_listen=ev,
        vad_kwargs={
            "min_silence_ms": 500,
            "min_speech_ms": 300,
            "thresh": 0.35,
        },
        STT_kwargs=stt_kw,
    )

    lines: list[str] = []
    for i in range(0, len(pcm), chunk_bytes):
        block = pcm[i : i + chunk_bytes]
        if len(block) < chunk_bytes:
            block = block + b"\x00" * (chunk_bytes - len(block))
        for line in pipe.process_audio_chunk(block):
            lines.append(line)

    pipe.reset_session()
    return lines
