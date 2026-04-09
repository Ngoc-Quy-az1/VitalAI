"""
Test STT (giọng nói → chữ) và/hoặc VAD + STT.

Chạy từ thư mục gốc repo (VitalAI):

  pip install -r requirements.txt

  # Chỉ STT trên file WAV (nhanh nhất)
  python scripts/test_speech.py --STT-only --wav "duong/dan/toi/file.wav"

  # VAD (Silero) rồi STT — file nên có tiếng + vài giây im lặng cuối
  python scripts/test_speech.py --vad-STT --wav "duong/dan/toi/file.wav"

  # Alias: --stt-only, --vad-stt, --asr-only, --vad-asr

Model Whisper mặc định là "tiny" (nhẹ). Đổi: --model small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from threading import Event

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_wav_mono(path: Path, target_sr: int = 16000):
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy()


def _float_to_pcm16_bytes(audio_f32: np.ndarray) -> bytes:
    clip = np.clip(audio_f32.astype(np.float32), -1.0, 1.0)
    pcm = (clip * 32767.0).astype(np.int16)
    return pcm.tobytes()


def run_STT_only(wav_path: Path, model: str) -> None:
    from src.STT.STT_handler import STTHandler

    handler = STTHandler()
    handler.setup(model_name=model)
    audio = _load_wav_mono(wav_path)
    text = handler.transcribe(audio, 16000)
    print(text or "(Không có văn bản)")


def run_vad_STT(wav_path: Path, model: str) -> None:
    from src.STT.vad_STT_pipeline import VADSTTPipeline

    audio = _load_wav_mono(wav_path)
    pcm = _float_to_pcm16_bytes(audio)
    chunk_samples = 512
    chunk_bytes = chunk_samples * 2

    ev = Event()
    ev.set()
    pipe = VADSTTPipeline()
    pipe.setup(
        should_listen=ev,
        vad_kwargs={
            "min_silence_ms": 500,
            "min_speech_ms": 300,
            "thresh": 0.35,
        },
        STT_kwargs={"model_name": model, "language": "vi"},
    )

    out: list[str] = []
    for i in range(0, len(pcm), chunk_bytes):
        block = pcm[i : i + chunk_bytes]
        if len(block) < chunk_bytes:
            block = block + b"\x00" * (chunk_bytes - len(block))
        for line in pipe.process_audio_chunk(block):
            out.append(line)

    pipe.reset_session()

    if not out:
        print(
            "VAD không trả về đoạn nào. Thử: file có tiếng rõ + vài giây im lặng cuối; "
            "hoặc chạy --STT-only để chỉ test Whisper."
        )
    else:
        for line in out:
            print(line)


def main() -> None:
    p = argparse.ArgumentParser(description="Test STT / VAD+STT (tiếng Việt)")
    p.add_argument("--wav", type=Path, required=True, help="File WAV (mono/stereo đều được)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--STT-only",
        "--stt-only",
        "--asr-only",
        dest="STT_only",
        action="store_true",
        help="Chỉ chạy Whisper trên cả file (bỏ qua VAD)",
    )
    g.add_argument(
        "--vad-STT",
        "--vad-stt",
        "--vad-asr",
        dest="vad_STT",
        action="store_true",
        help="Chunk qua Silero VAD rồi Whisper",
    )
    p.add_argument(
        "--model",
        default="tiny",
        help='Tên model Whisper: tiny, base, small, ... (mặc định: tiny)',
    )
    args = p.parse_args()

    if not args.wav.is_file():
        sys.exit(f"Không tìm thấy file: {args.wav}")

    if args.STT_only:
        run_STT_only(args.wav, args.model)
    else:
        run_vad_STT(args.wav, args.model)


if __name__ == "__main__":
    main()
