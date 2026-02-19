import re
import time

HALLUCINATION_PATTERNS = [
    r"^\[.*\]$",
    r"^\(.*\)$",
    r"^\*.*\*$",
    # r"^-\s*(yeah|yes|no|ok)\.$",
    r"^(music|musique|bruit|silence|applause|noise)$",
    # r"^(merci|thank you|thanks|thank you\.|thanks\.)$",
    # r"^(you|we)['']re\s+not\s+sure.*$",
    r"^sous-titres.*$",
    r"^(sous-titres réalisés|transcribed by|subtitles by).*$",
]

REPEAT_WORD_MIN = 3  # catch loops after 3 repetitions, not 4


def _detect_repetition_loop(text: str) -> bool:
    clean = re.sub(r"[^\w\s']", " ", text.lower())
    words = clean.split()

    if len(words) < 4:
        return False

    for window in range(1, 7):
        if len(words) < window * REPEAT_WORD_MIN:
            continue
        phrase = tuple(words[:window])
        count = 1
        i = window
        while i + window <= len(words):
            if tuple(words[i: i + window]) == phrase:
                count += 1
                i += window
                if count >= REPEAT_WORD_MIN:
                    return True
            else:
                break

    half = len(words) // 2
    if half >= 3:
        first = " ".join(words[:half])
        second = " ".join(words[half: half * 2])
        if first.lower() == second.lower():
            return True

    return False


def is_hallucination(text: str) -> bool:
    t = text.strip()
    for pattern in HALLUCINATION_PATTERNS:
        if re.match(pattern, t, re.IGNORECASE):
            return True
    if _detect_repetition_loop(t):
        return True
    return False


class WhisperASR:
    def __init__(self, model_size: str = "tiny"):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed.\nRun: pip install faster-whisper"
            )

        print(f"⏳ Loading faster-whisper [{model_size}] on CPU (int8)...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.model_size = model_size
        print(f"✅ WhisperASR ready (faster-whisper/{model_size}/cpu/int8)")

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        import os

        if not os.path.exists(audio_path):
            return "[file_not_found]"
        if os.path.getsize(audio_path) < 1000:
            return "[file_too_small]"

        t0 = time.time()
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                temperature=[0.0, 0.2, 0.4],
                compression_ratio_threshold=1.8,
                log_prob_threshold=-0.8,
                no_speech_threshold=0.5,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 200,
                    "speech_pad_ms": 100,
                },

                beam_size=1,
            )

            parts = [seg.text.strip() for seg in segments if seg.text.strip()]
            text = " ".join(parts).strip()
            elapsed = time.time() - t0
            print(f"[ASR] {elapsed:.1f}s → '{text[:80]}'")

            if not text:
                return "[silence]"

            if is_hallucination(text):
                print(f"[ASR] Filtered hallucination: '{text[:60]}'")
                return "[hallucination]"

            return text

        except Exception as e:
            print(f"[ASR] Exception ({type(e).__name__}): {e}")
            return "[error]"