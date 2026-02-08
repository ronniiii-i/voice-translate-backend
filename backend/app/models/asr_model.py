
import subprocess
import json
import os
from pathlib import Path

class WhisperASR:
    def __init__(self, model_size="tiny"):
        self.root_dir = Path(__file__).parent.parent.parent.parent
        self.whisper_bin = self.root_dir / "models/asr/whisper.cpp/build/bin/whisper-cli"        
        self.model_path = self.root_dir / f"models/asr/whisper.cpp/models/ggml-{model_size}.bin"
        
        if not self.whisper_bin.exists():
            raise FileNotFoundError(f"Whisper binary not found at {self.whisper_bin}. Did you run 'make'?")

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """Transcribe audio file to text using whisper.cpp"""
        
        json_file = Path(f"{audio_path}.json")
        
        cmd = [
            str(self.whisper_bin),
            "-m", str(self.model_path),
            "-f", str(audio_path),
            "-l", language,
            "-nt",  
            "-oj"   
        ]
        
        try:
            if json_file.exists():
                json_file.unlink()

            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Wait a split second for the OS to flush the file to disk
            import time
            time.sleep(0.1)

            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                json_file.unlink()
                
                text = " ".join([seg['text'] for seg in data.get('transcription', [])])
                return text.strip()
            else:
                alt_json = Path(str(audio_path).replace(".wav", "") + ".json")
                if alt_json.exists():
                    pass
                return "Error: Whisper output file missing."
        except subprocess.CalledProcessError as e:
            return f"Error during transcription: {e.stderr}"