# backend/app/models/tts_model.py
import subprocess
import shlex
from pathlib import Path

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

    def synthesize(self, text: str, output_path: str, language: str = "en") -> str:
        """Convert text to speech using Piper"""
        model_name = self.model_map.get(language, self.model_map["en"])
        model_path = self.model_dir / model_name

        if not model_path.exists():
            print(f"[TTS] ⚠️ Model not found: {model_path}, falling back to English")
            model_path = self.model_dir / self.model_map["en"]

        # FIX: Use list form + stdin to avoid shell injection with special chars
        cmd = ["piper", "--model", str(model_path), "--output_file", output_path]
        
        try:
            result = subprocess.run(
                cmd,
                input=text,          # Pass text via stdin (safe — no shell escaping needed)
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"[TTS] ❌ Error: {e.stderr[:100] if e.stderr else str(e)}")
            raise
        except subprocess.TimeoutExpired:
            print(f"[TTS] ❌ Timeout after 30s")
            raise