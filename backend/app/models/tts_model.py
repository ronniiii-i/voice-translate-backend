import subprocess
from pathlib import Path

class PiperTTS:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.parent.parent
        self.model_dir = self.root_dir / "models/tts"
        
    def synthesize(self, text: str, output_path: str, language: str = "en") -> str:
        """Convert text to speech"""
        model_map = {
            "en": "en_US-bryce-medium.onnx",
            "fr": "fr_FR-siwis-medium.onnx",
            "es": "en_US-bryce-medium.onnx" 
        }
        
        model_path = self.model_dir / model_map.get(language, model_map["en"])
        
        # Run Piper
        cmd = f'echo "{text}" | piper --model {model_path} --output_file {output_path}'
        subprocess.run(cmd, shell=True, check=True)
        
        return output_path
