# backend/app/services/translation_pipeline.py
from app.models.asr_model import WhisperASR
from app.models.mt_model import ArgosTranslator
from app.models.tts_model import PiperTTS
import tempfile
import os
import time

class TranslationPipeline:
    def __init__(self):
        self.asr = WhisperASR(model_size="tiny")  # tiny = 2-4s on your CPU
        self.translator = ArgosTranslator(warmup_pairs=[
            ("en", "fr"), ("fr", "en"),
            ("en", "es"), ("es", "en"),
            ("en", "de"), ("de", "en"),
        ])
        self.tts = PiperTTS()
        print("✅ Pipeline ready (tiny model)")

    def process_audio(self, audio_path: str, source_lang: str, target_lang: str) -> tuple:
        """Audio → Text → Translation → Audio"""

        # Step 1: ASR
        t0 = time.time()
        source_text = self.asr.transcribe(audio_path, language=source_lang)
        print(f"[ASR] {time.time()-t0:.1f}s → '{source_text[:60]}'")

        # Skip silence / errors
        if not source_text or source_text.startswith("["):
            print(f"[Pipeline] Skipping — no usable speech")
            return None, source_text, "[no translation]"

        # Step 2: MT
        t1 = time.time()
        translated_text = self.translator.translate(source_text, source_lang, target_lang)
        print(f"[MT] {time.time()-t1:.1f}s  '{source_lang}→{target_lang}': '{translated_text[:60]}'")

        # Step 3: TTS
        t2 = time.time()
        out_audio = tempfile.mktemp(suffix=".wav")
        self.tts.synthesize(translated_text, out_audio, language=target_lang)
        print(f"[TTS] {time.time()-t2:.1f}s → {out_audio}")

        total = time.time() - t0
        print(f"[Pipeline] Total: {total:.1f}s")

        return out_audio, source_text, translated_text