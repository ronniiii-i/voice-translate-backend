# backend/app/models/asr_model.py
import re
import time

# Whisper hallucinates these when it hears music/noise/silence
# Filter them out so they don't go through MT/TTS
HALLUCINATION_PATTERNS = [
    r"^\[.*\]$",                        # [Music], [Applause], [Silence]
    r"^\(.*\)$",                        # (musique), (coughs), (upbeat music)
    r"^\*.*\*$",                        # *Mais de l'Ontario*
    r"^-\s*(yeah|yes|no)\.$",
    r"^(merci|thank you|thanks)\.$",
    r"^(music|musique|bruit|silence|applause)$",
    r"^(you|we)['']re\s+not\s+sure.*$",  # "We're not sure..." loops
]

def is_hallucination(text: str) -> bool:
    """Return True if Whisper is hallucinating background noise, not real speech."""
    t = text.strip()
    for pattern in HALLUCINATION_PATTERNS:
        if re.match(pattern, t, re.IGNORECASE):
            return True
    # Repeated phrases = Whisper stuck in a loop on noise
    words = t.split()
    if len(words) >= 6:
        half = " ".join(words[: len(words) // 2])
        second = " ".join(words[len(words) // 2 :])
        if half.lower() == second.lower():
            return True
    return False


class WhisperASR:
    def __init__(self, model_size: str = "tiny"):
        """
        Uses faster-whisper — pure Python, no subprocess, no file output race conditions.

        Model sizing for your Core i5 CPU:
          tiny  → ~2-4s per 5s audio  ✅ USE THIS
          base  → ~8-15s per 5s audio ❌ too slow
          small → way too slow on CPU ❌
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed.\n"
                "Run: pip install faster-whisper"
            )

        print(f"⏳ Loading faster-whisper [{model_size}] ...")
        # cpu + int8 = fastest possible on your laptop
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.model_size = model_size
        print(f"✅ WhisperASR ready (faster-whisper/{model_size}/int8)")

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe audio file. Returns transcribed text or a bracketed
        sentinel like '[silence]', '[error]' etc. that the pipeline skips.
        """
        import os

        if not os.path.exists(audio_path):
            return "[file_not_found]"
        if os.path.getsize(audio_path) < 1000:
            return "[file_too_small]"

        t0 = time.time()
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,          # avoids language detection overhead
                beam_size=5,
                best_of=5,
                temperature=0.0,            # greedy — fastest + most deterministic
                vad_filter=True,            # built-in VAD: skips silent chunks
                vad_parameters={
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 200,
                },
                no_speech_threshold=0.6,    # skip low-confidence segments
                log_prob_threshold=-1.0,    # skip very uncertain segments
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,  # prevents repetition loops
            )

            parts = []
            for seg in segments:
                t = seg.text.strip()
                if t:
                    parts.append(t)

            text = " ".join(parts).strip()
            elapsed = time.time() - t0
            print(f"[ASR] {elapsed:.1f}s → '{text[:80]}'")

            if not text:
                return "[silence]"

            if is_hallucination(text):
                print(f"[ASR] Filtered hallucination: '{text[:50]}'")
                return "[hallucination]"

            return text

        except Exception as e:
            print(f"[ASR] Exception: {e}")
            return "[error]"