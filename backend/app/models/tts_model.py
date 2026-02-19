# backend/app/models/tts_model.py
import subprocess
from pathlib import Path
import os
import time

class PiperTTS:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.parent.parent
        self.model_dir = self.root_dir / "models/tts"

        self.model_map = {
            "en": "en_US-bryce-medium.onnx",
            "fr": "fr_FR-siwis-medium.onnx",
            "de": "de_DE-thorsten-medium.onnx",
            "es": "es_ES-mls_10246-low.onnx",
        }

        # Verify models exist at startup, warn if missing
        for lang, fname in self.model_map.items():
            p = self.model_dir / fname
            if not p.exists():
                print(f"[TTS] ⚠️  Missing model: {p}")

    def _get_model_path(self, language: str) -> Path:
        fname = self.model_map.get(language, self.model_map["en"])
        path = self.model_dir / fname
        if not path.exists():
            # Fall back to English
            print(f"[TTS] ⚠️  {language} model missing, using English")
            path = self.model_dir / self.model_map["en"]
        return path

    def synthesize(self, text: str, output_path: str, language: str = "en") -> str:
        """
        Convert text to speech using Piper.
        Text is passed via stdin to avoid shell-injection issues with
        apostrophes and special characters in translated text.
        """
        model_path = self._get_model_path(language)
        t0 = time.time()

        cmd = [
            "piper",
            "--model", str(model_path),
            "--output_file", output_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                input=text,           # stdin — safe for any text content
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            elapsed = time.time() - t0
            if elapsed > 10:
                print(f"[TTS] ⚠️  Slow synthesis: {elapsed:.1f}s for '{text[:40]}'")
            return output_path

        except subprocess.CalledProcessError as e:
            err = e.stderr.strip()[:200] if e.stderr else str(e)
            print(f"[TTS] ❌ Error: {err}")
            raise
        except subprocess.TimeoutExpired:
            print(f"[TTS] ❌ Timeout after 30s")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "Piper not found. Install it:\n"
                "  pip install piper-tts\n"
                "  OR download from https://github.com/rhasspy/piper/releases"
            )