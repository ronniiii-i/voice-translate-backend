from app.models.asr_model import WhisperASR
from app.models.mt_model import ArgosTranslator
from app.models.tts_model import PiperTTS
import tempfile
import os

class TranslationPipeline:
    def __init__(self):
        self.asr = WhisperASR()
        self.translator = ArgosTranslator(warmup_pairs=[("en", "fr"), ("fr", "en")])
        self.tts = PiperTTS()
        print("Warming up models...")
        self.translator.translate("warm up", "en", "fr")
        
        
    def process_audio(self, audio_path: str, source_lang: str, target_lang: str) -> str:
        """Full pipeline: Audio → Text → Translation → Audio"""
        
        # Step 1: Speech to Text
        print(f"[ASR] Transcribing {source_lang} audio...")
        source_text = self.asr.transcribe(audio_path, language=source_lang)
        print(f"[ASR] Result: {source_text}")
        
        # Step 2: Translate
        print(f"[MT] Translating {source_lang} → {target_lang}...")
        translated_text = self.translator.translate(source_text, source_lang, target_lang)
        print(f"[MT] Result: {translated_text}")
        
        # Step 3: Text to Speech
        print(f"[TTS] Synthesizing {target_lang} audio...")
        output_audio = tempfile.mktemp(suffix=".wav")
        self.tts.synthesize(translated_text, output_audio, language=target_lang)
        print(f"[TTS] Saved to: {output_audio}")
        
        return output_audio, source_text, translated_text