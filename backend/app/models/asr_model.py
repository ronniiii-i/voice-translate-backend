# backend/app/models/asr_model.py
import subprocess
import json
import os
import re
from pathlib import Path
import time

# Whisper hallucinates these when it hears music/noise/silence
# Filter them out so they don't go through MT/TTS
HALLUCINATION_PATTERNS = [
    r"^\[.*\]$",           # [Music], [Applause], [Silence]
    r"^\(.*\)$",           # (musique), (coughs), (upbeat music)
    r"^\*.*\*$",           # *Mais de l'Ontario*
    r"^-\s*(yeah|yes|no)\.$",  # - Yeah. - Yeah.
    r"^(merci|thank you|thanks)\.$",
    r"^(music|musique|bruit|silence|applause)$",
]

def is_hallucination(text: str) -> bool:
    """Return True if Whisper is hallucinating background noise, not real speech."""
    t = text.strip()
    for pattern in HALLUCINATION_PATTERNS:
        if re.match(pattern, t, re.IGNORECASE):
            return True
    # Repeated phrases = Whisper stuck in a loop on noise
    # e.g. "I'm not sure. I'm not sure. I'm not sure."
    words = t.split()
    if len(words) >= 6:
        half = " ".join(words[:len(words)//2])
        second = " ".join(words[len(words)//2:])
        if half.lower() == second.lower():
            return True
    return False


class WhisperASR:
    def __init__(self, model_size="tiny"):
        """
        tiny: 2-4s per 5s audio — USE THIS for real-time on your CPU
        base: 9-15s per 5s audio — too slow, causes cascading timeouts
        """
        self.root_dir = Path(__file__).parent.parent.parent.parent
        self.whisper_bin = self.root_dir / "models/asr/whisper.cpp/build/bin/whisper-cli"
        self.model_path = self.root_dir / f"models/asr/whisper.cpp/models/ggml-{model_size}.bin"

        if not self.whisper_bin.exists():
            raise FileNotFoundError(f"Whisper binary not found: {self.whisper_bin}")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Run: cd models/asr/whisper.cpp && "
                f"bash ./models/download-ggml-model.sh {model_size}"
            )
        print(f"✅ WhisperASR: {model_size} model ready")

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        if not os.path.exists(audio_path):
            return "[file_not_found]"
        if os.path.getsize(audio_path) < 1000:
            return "[file_too_small]"

        json_file = Path(f"{audio_path}.json")

        cmd = [
            str(self.whisper_bin),
            "-m", str(self.model_path),
            "-f", str(audio_path),
            "-l", language,
            "-t", "4",
            "-oj",
            "-nt",
            "--temperature", "0.0",
            "--beam-size", "5",
            "--no-speech-thold", "0.6",  # Skip segments with low speech probability
            "--entropy-thold", "2.8",    # Skip high-entropy (uncertain) segments
        ]

        try:
            if json_file.exists():
                json_file.unlink()

            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20)
            time.sleep(0.05)

            if not json_file.exists():
                return "[no_output]"

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            json_file.unlink()

            parts = [
                seg["text"].strip()
                for seg in data.get("transcription", [])
                if seg.get("text", "").strip()
            ]
            text = " ".join(parts).strip()

            if not text:
                return "[silence]"

            # Filter Whisper hallucinations (music, noise, repeating loops)
            if is_hallucination(text):
                print(f"[ASR] Filtered hallucination: '{text[:50]}'")
                return "[hallucination]"

            return text

        except subprocess.TimeoutExpired:
            print(f"[ASR] Timeout — try tiny model if using base")
            return "[timeout]"
        except subprocess.CalledProcessError as e:
            print(f"[ASR] Error: {e.stderr[:100] if e.stderr else str(e)}")
            return "[error]"
        except Exception as e:
            print(f"[ASR] Exception: {e}")
            return "[error]"