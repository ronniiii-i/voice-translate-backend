# backend/app/services/translation_pipeline.py
from app.models.asr_model import WhisperASR
from app.models.mt_model import ArgosTranslator
from app.models.tts_model import PiperTTS
import tempfile
import os
import time


class TranslationPipeline:
    def __init__(self):
        # tiny = 2-4s on i5 CPU with faster-whisper int8
        self.asr = WhisperASR(model_size="tiny")
        self.translator = ArgosTranslator(warmup_pairs=[
            ("en", "fr"), ("fr", "en"),
            ("en", "es"), ("es", "en"),
            ("en", "de"), ("de", "en"),
        ])
        self.tts = PiperTTS()
        print("✅ Translation Pipeline ready")

    def process_audio(self, audio_path: str, source_lang: str, target_lang: str) -> tuple:
        """
        Full pipeline: WAV file → transcription → translation → TTS audio file.

        Returns:
            (output_wav_path, source_text, translated_text)
            output_wav_path may be None if ASR finds no speech.
        """
        t_total = time.time()

        # ── Step 1: ASR ──────────────────────────────────────────────────────
        t0 = time.time()
        source_text = self.asr.transcribe(audio_path, language=source_lang)
        print(f"[ASR] {time.time()-t0:.1f}s → '{source_text[:80]}'")

        # Bracketed sentinels mean no usable speech — skip MT+TTS
        if not source_text or source_text.startswith("["):
            print("[Pipeline] No usable speech, skipping MT+TTS")
            return None, source_text, "[no translation]"

        # ── Step 2: MT ───────────────────────────────────────────────────────
        t1 = time.time()
        try:
            translated_text = self.translator.translate(source_text, source_lang, target_lang)
        except Exception as e:
            print(f"[MT] Error: {e}")
            return None, source_text, "[translation error]"
        print(f"[MT] {time.time()-t1:.1f}s  '{source_lang}→{target_lang}': '{translated_text[:80]}'")

        # ── Step 3: TTS ──────────────────────────────────────────────────────
        t2 = time.time()
        try:
            fd, out_audio = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            self.tts.synthesize(translated_text, out_audio, language=target_lang)
        except Exception as e:
            print(f"[TTS] Error: {e}")
            return None, source_text, translated_text  # still return text even if audio fails
        print(f"[TTS] {time.time()-t2:.1f}s → {out_audio}")

        print(f"[Pipeline] Total: {time.time()-t_total:.1f}s")
        return out_audio, source_text, translated_text