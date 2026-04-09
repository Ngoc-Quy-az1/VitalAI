"""
VADIterator — luồng giả lập theo Silero VAD.
Tham khảo: https://github.com/snakers4/silero-vad
"""

import torch


class VADIterator:
    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        """
        Parameters
        ----------
        model: preloaded .jit/.onnx silero VAD model

        threshold: float (default - 0.5)
            Ngưỡng xác suất tiếng nói.

        sampling_rate: int (default - 16000)
            Silero hỗ trợ 8000 và 16000.

        min_silence_duration_ms: int (default - 100)
            Khoảng lặng tối thiểu trước khi kết thúc một đoạn nói.

        speech_pad_ms: int (default - 30)
            Đệm hai đầu đoạn nói (ms).
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.is_speaking = False
        self.buffer = []

        if sampling_rate not in [8000, 16000]:
            raise ValueError(
                "VADIterator does not support sampling rates other than [8000, 16000]"
            )

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x):
        """
        x: torch.Tensor — chunk audio (xem ví dụ trong repo silero-vad)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except Exception as e:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually") from e

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            return None

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                self.temp_end = 0
                self.triggered = False
                spoken_utterance = self.buffer
                self.buffer = []
                return spoken_utterance

        if self.triggered:
            self.buffer.append(x)

        return None
