# backend/app/services/vad_processor.py
import struct
import time
import math
from collections import deque


class StreamingVAD:
    """
    VAD with adaptive energy threshold and max duration cap.
    Tuned for your environment based on the energy logs (avg ~100-150 during silence).
    """

    def __init__(
        self,
        silence_threshold: float = 1.2,      # seconds of silence before triggering
        min_speech_duration: float = 0.5,    # ignore segments shorter than this
        energy_threshold: float = 500.0,     # RMS threshold — tune this for your mic
        max_speech_duration: float = 8.0,    # force process after this many seconds
    ):
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.energy_threshold = energy_threshold
        self.max_speech_duration = max_speech_duration

        self.last_speech_time = None
        self.speech_start_time = None
        self.audio_buffer = []

        self._energy_log = deque(maxlen=20)
        self._log_counter = 0

    def _rms(self, chunk: bytes) -> float:
        if len(chunk) < 2:
            return 0.0
        try:
            n = len(chunk) // 2
            samples = struct.unpack(f"{n}h", chunk)
            return math.sqrt(sum(s * s for s in samples) / n)
        except:
            return 0.0

    def is_silence(self, chunk: bytes) -> bool:
        energy = self._rms(chunk)
        self._energy_log.append(energy)

        # Log average energy periodically for threshold tuning
        self._log_counter += 1
        if self._log_counter % 50 == 0 and self._energy_log:
            avg = sum(self._energy_log) / len(self._energy_log)
            speaking = "SPEAKING" if self.speech_start_time else "silence"
            print(f"[VAD] Avg energy: {avg:.0f} (threshold: {self.energy_threshold}) [{speaking}]")

        return energy < self.energy_threshold

    def add_chunk(self, chunk: bytes) -> tuple[bool, bytes]:
        now = time.time()
        silent = self.is_silence(chunk)

        if not silent:
            if self.speech_start_time is None:
                self.speech_start_time = now
                print("[VAD] 🎤 Speech started")
            self.last_speech_time = now
            self.audio_buffer.append(chunk)

            # Hard cap — prevents >8s segments reaching Whisper
            if now - self.speech_start_time >= self.max_speech_duration:
                print(f"[VAD] ⏱ Max duration ({self.max_speech_duration}s) reached, processing")
                return self._emit()

        else:
            if self.last_speech_time is not None:
                silence_dur = now - self.last_speech_time
                if silence_dur >= self.silence_threshold:
                    speech_dur = self.last_speech_time - (self.speech_start_time or self.last_speech_time)
                    if speech_dur >= self.min_speech_duration:
                        print(f"[VAD] 🔇 Silence detected after {speech_dur:.2f}s of speech")
                        return self._emit()
                    else:
                        print(f"[VAD] Too short ({speech_dur:.2f}s), discarding")
                        self._reset()

        return False, b""

    def _emit(self) -> tuple[bool, bytes]:
        data = b"".join(self.audio_buffer)
        self._reset()
        return True, data

    def _reset(self):
        self.last_speech_time = None
        self.speech_start_time = None
        self.audio_buffer = []