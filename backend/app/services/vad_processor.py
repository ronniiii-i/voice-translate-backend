# backend/app/services/vad_processor.py
import struct
import time
import math
from collections import deque


class StreamingVAD:
    """
    VAD with adaptive energy threshold and max duration cap.

    Calibration guide:
      - Run the server, speak normally, watch the [VAD] Avg energy logs.
      - If avg SILENCE is ~100-200 and avg SPEECH is ~800-2000, keep threshold=500.
      - If your mic is quiet and speech only hits ~300, lower to 250.
      - If you're in a noisy room and silence is ~400, raise to 600.
    """

    def __init__(
        self,
        silence_threshold: float = 1.2,       # seconds of silence before triggering
        min_speech_duration: float = 0.4,     # ignore very short blips
        energy_threshold: float = 400.0,      # RMS threshold — lower = more sensitive
        max_speech_duration: float = 8.0,     # force-process after this many seconds
        sample_rate: int = 16000,
        sample_width: int = 2,                # bytes per sample (Int16 = 2)
    ):
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.energy_threshold = energy_threshold
        self.max_speech_duration = max_speech_duration
        self.sample_rate = sample_rate
        self.sample_width = sample_width

        self.last_speech_time = None
        self.speech_start_time = None
        self.audio_buffer = []

        self._energy_log = deque(maxlen=30)
        self._log_counter = 0
        self._chunks_received = 0

    def _rms(self, chunk: bytes) -> float:
        """Compute Root Mean Square energy of a PCM Int16 chunk."""
        n = len(chunk) // self.sample_width
        if n == 0:
            return 0.0
        try:
            samples = struct.unpack(f"<{n}h", chunk)  # little-endian Int16
            return math.sqrt(sum(s * s for s in samples) / n)
        except struct.error:
            return 0.0

    def is_speech(self, chunk: bytes) -> bool:
        """Return True if this chunk contains speech-level energy."""
        energy = self._rms(chunk)
        self._energy_log.append(energy)

        self._log_counter += 1
        # Log every 30 chunks (~3s at 4096-sample chunks) so you can calibrate
        if self._log_counter % 30 == 0 and self._energy_log:
            avg = sum(self._energy_log) / len(self._energy_log)
            peak = max(self._energy_log)
            state = "🎤 SPEAKING" if self.speech_start_time else "🔇 silence"
            print(
                f"[VAD] avg={avg:.0f} peak={peak:.0f} "
                f"threshold={self.energy_threshold:.0f} [{state}]"
            )

        return energy >= self.energy_threshold

    def add_chunk(self, chunk: bytes) -> tuple:
        """
        Process one raw PCM chunk.
        Returns (should_process: bool, audio_bytes: bytes)
        """
        self._chunks_received += 1
        now = time.time()
        speaking = self.is_speech(chunk)

        if speaking:
            if self.speech_start_time is None:
                self.speech_start_time = now
                print("[VAD] 🎤 Speech started")
            self.last_speech_time = now
            self.audio_buffer.append(chunk)

            # Hard cap to prevent Whisper from receiving huge segments
            speech_dur = now - self.speech_start_time
            if speech_dur >= self.max_speech_duration:
                print(f"[VAD] ⏱ Max duration ({self.max_speech_duration}s) reached — processing")
                return self._emit()

        else:
            # Silence
            if self.speech_start_time is not None and self.last_speech_time is not None:
                silence_dur = now - self.last_speech_time
                if silence_dur >= self.silence_threshold:
                    speech_dur = self.last_speech_time - self.speech_start_time
                    if speech_dur >= self.min_speech_duration:
                        print(
                            f"[VAD] 🔇 Silence after {speech_dur:.2f}s of speech — sending to ASR"
                        )
                        return self._emit()
                    else:
                        print(f"[VAD] Too short ({speech_dur:.2f}s), discarding")
                        self._reset()

        return False, b""

    def _emit(self) -> tuple:
        data = b"".join(self.audio_buffer)
        self._reset()
        return True, data

    def _reset(self):
        self.last_speech_time = None
        self.speech_start_time = None
        self.audio_buffer = []