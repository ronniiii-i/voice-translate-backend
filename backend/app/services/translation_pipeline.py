from app.models.asr_model import WhisperASR
from app.models.mt_model import HelsinkiTranslator
from app.models.tts_model import PiperTTS
import tempfile
import os
import time


class TranslationPipeline:
    def __init__(self):
        self.asr = WhisperASR(model_size="base")
        self.translator = HelsinkiTranslator()
        self.tts = PiperTTS()
        print("✅ Translation Pipeline ready")

    def process_audio(
        self,
        audio_path: str,
        source_lang: str,
        target_lang: str,
        context: list[str] | None = None,
    ) -> tuple:
        t_total = time.time()

        # ── Step 1: ASR ──────────────────────────────────────────────────────
        t0 = time.time()
        source_text = self.asr.transcribe(audio_path, language=source_lang)
        print(f"[ASR] {time.time() - t0:.1f}s → '{source_text[:80]}'")

        # Bracketed sentinels mean no usable speech — skip MT + TTS
        if not source_text or source_text.startswith("["):
            print("[Pipeline] No usable speech, skipping MT+TTS")
            return None, source_text, "[no translation]"

        # ── Step 2: MT (context-aware) ───────────────────────────────────────
        t1 = time.time()
        try:
            translated_text = self.translator.translate(
                source_text,
                source_lang,
                target_lang,
                context=context,
                use_context=True,
            )
        except Exception as e:
            print(f"[MT] Error: {e}")
            return None, source_text, "[translation error]"
        print(f"[MT] {time.time() - t1:.1f}s  "
              f"'{source_lang}→{target_lang}': '{translated_text[:80]}'")

        # ── Step 3: TTS ──────────────────────────────────────────────────────
        t2 = time.time()
        try:
            fd, out_audio = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            self.tts.synthesize(translated_text, out_audio, language=target_lang)
        except Exception as e:
            print(f"[TTS] Error: {e}")
            # Return text even if audio fails — frontend can show caption only
            return None, source_text, translated_text
        print(f"[TTS] {time.time() - t2:.1f}s → {out_audio}")

        print(f"[Pipeline] Total: {time.time() - t_total:.1f}s")
        return out_audio, source_text, translated_text