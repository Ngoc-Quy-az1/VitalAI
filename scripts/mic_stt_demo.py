"""
Demo mic + VAD + STT (faster-whisper). Chạy: python scripts/mic_stt_demo.py
"""

from __future__ import annotations

import queue
import sys
import warnings

# Giam canh bao requests/urllib3 khong khop phien ban (torch.hub)
warnings.filterwarnings("ignore", message=".*supported version.*")
import threading
from pathlib import Path
from threading import Event

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sounddevice as sd

from src.STT.vad_STT_pipeline import VADSTTPipeline

SAMPLE_RATE = 16000
# Silero VAD 16 kHz: mỗi bước 512 mẫu
VAD_CHUNK_SAMPLES = 512


class MicSTTDemo:
    def __init__(self) -> None:
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title("VitalAI — STT theo từng câu (VAD + Whisper)")
        self.root.geometry("640x460")
        self.root.minsize(480, 320)

        self.running = False
        self.stream: sd.InputStream | None = None
        self.audio_q: queue.Queue[bytes] = queue.Queue(maxsize=256)
        self.text_q: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.should_listen = Event()
        self.pipe: VADSTTPipeline | None = None

        self.btn = tk.Button(
            self.root,
            text="Bật mic",
            command=self.toggle,
            width=22,
            height=2,
            font=("Segoe UI", 11),
        )
        self.btn.pack(pady=10)

        self.lbl = tk.Label(
            self.root,
            text="Mic đang tắt — bấm «Bật mic», nói từng câu rồi ngắt nhịp",
            fg="gray",
            font=("Segoe UI", 10),
        )
        self.lbl.pack()

        self.txt = tk.Text(
            self.root,
            height=16,
            wrap="word",
            font=("Segoe UI", 11),
            padx=8,
            pady=8,
        )
        self.txt.pack(fill="both", expand=True, padx=12, pady=8)

        self.hint = tk.Label(
            self.root,
            text="faster-whisper (small, int8 trên CPU) + VAD. Ngắt ~0,4s giữa câu. GPU: nhanh hơn rõ.",
            fg="gray",
            font=("Segoe UI", 9),
        )
        self.hint.pack(pady=(0, 8))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _ensure_pipeline(self) -> None:
        if self.pipe is not None:
            return
        self.should_listen.set()
        self.pipe = VADSTTPipeline()
        self.pipe.setup(
            should_listen=self.should_listen,
            vad_kwargs={
                "min_silence_ms": 400,
                "min_speech_ms": 300,
                "thresh": 0.45,
            },
            STT_kwargs={
                "engine": "faster",
                "model_name": "small",
                "language": "vi",
                "beam_size": 3,
            },
        )

    def toggle(self) -> None:
        if not self.running:
            self._start_mic()
        else:
            self._stop_mic()

    def _audio_callback(self, indata, frames, t, status) -> None:
        if status:
            print(status, file=sys.stderr)
        if not self.running:
            return
        pcm = np.clip(indata[:, 0], -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        block = pcm_i16.tobytes()
        try:
            self.audio_q.put_nowait(block)
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        assert self.pipe is not None
        while self.running:
            try:
                chunk = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                for line in self.pipe.process_audio_chunk(chunk):
                    self.text_q.put(line)
            except Exception as e:
                self.text_q.put(f"\n[Lỗi: {e}]\n")

    def _poll_text(self) -> None:
        while True:
            try:
                line = self.text_q.get_nowait()
            except queue.Empty:
                break
            self.txt.insert(self._tk.END, line.strip() + "\n")
            self.txt.see(self._tk.END)
        if self.running:
            self.root.after(60, self._poll_text)

    def _start_mic(self) -> None:
        self._ensure_pipeline()
        self.running = True
        self.btn.config(text="Tắt mic")
        self.lbl.config(text="Đang nghe — nói từng câu, ngắt nhịp giữa các câu", fg="green")
        self.txt.delete("1.0", self._tk.END)

        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break
        while not self.text_q.empty():
            try:
                self.text_q.get_nowait()
            except queue.Empty:
                break

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=VAD_CHUNK_SAMPLES,
            callback=self._audio_callback,
        )
        self.stream.start()

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        self.root.after(60, self._poll_text)

    def _stop_mic(self) -> None:
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.pipe is not None:
            self.pipe.reset_session()
        self.should_listen.set()
        self.btn.config(text="Bật mic")
        self.lbl.config(
            text="Mic đang tắt — bấm «Bật mic», nói từng câu rồi ngắt nhịp",
            fg="gray",
        )

    def on_close(self) -> None:
        self._stop_mic()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    try:
        app = MicSTTDemo()
        print(
            "Da mo cua so GUI — neu khong thay, kiem tra taskbar / man hinh khac.",
            flush=True,
        )
        app.run()
    except Exception:
        import traceback

        traceback.print_exc()
        print(
            "\nNeu loi _tkinter: cai Python day du (tcl/tk) hoac chay pythonw.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
