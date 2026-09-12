"""
eval_wer.py — Table 1: ASR Word Error Rate per language (Revised Baseline)

HOW TO RUN:
    From your backend/ folder:
        python evaluation/eval_wer.py

WHAT IT DOES:
    - Downloads 100 utterances per language from FLEURS (google/fleurs) with
      Common Voice as supplement.
    - Transcribes each using faster-whisper base (CPU int8).
    - Applies revised normalization:
        * OpenCC for complete Traditional->Simplified Chinese conversion.
        * Hyphen-to-space replacement to prevent false compound word errors.
        * Standard lowercasing and punctuation stripping.
    - Computes corpus-level WER against reference transcripts using jiwer.
    - Saves results to evaluation/results_wer.txt
"""

import os
import sys
import string
import jiwer
import opencc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.asr_model import WhisperASR
from eval_datasets import get_asr_samples

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results_wer.txt")

LANGUAGES        = ["en", "fr", "de", "es", "zh"]
SAMPLES_PER_LANG = 100

# Full Chinese Traditional to Simplified converter
zh_converter = opencc.OpenCC('t2s.json')


def normalise_zh(text: str) -> str:
    # Convert traditional characters to simplified accurately
    text = zh_converter.convert(text)
    zh_punct = "。，、？！；：\"''（）【】《》…—～·"
    text = text.translate(str.maketrans("", "", zh_punct + string.punctuation))
    return " ".join(list(text.strip()))


def normalise(text: str, lang: str = "") -> str:
    if lang == "zh":
        return normalise_zh(text)
    
    text = text.lower().strip()
    # Replace hyphens with spaces so hyphenated words don't merge into a single word
    text = text.replace("-", " ")
    # Strip remaining punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Standardize whitespace
    return " ".join(text.split())


def transcribe_zh(asr, filepath: str) -> str:
    import time
    t0 = time.time()
    try:
        segments, _ = asr.model.transcribe(
            filepath,
            language="zh",
            initial_prompt="以下是普通话的转录。",
            temperature=[0.0, 0.2, 0.4],
            compression_ratio_threshold=1.8,
            log_prob_threshold=-0.8,
            no_speech_threshold=0.5,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200, "speech_pad_ms": 100},
            beam_size=1,
        )
        parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        text  = "".join(parts).strip()
        print(f"    [{time.time()-t0:.1f}s] → '{text[:60]}'")
        return text if text else "[silence]"
    except Exception as e:
        return f"[error: {e}]"


def main():
    print("Loading WhisperASR (base model)...")
    asr = WhisperASR(model_size="base")
    print("Model loaded.\n")

    all_results = {}

    for lang in LANGUAGES:
        print(f"\n── {lang.upper()} ──────────────────────────────────────────")
        print(f"Fetching {SAMPLES_PER_LANG} samples from FLEURS/Common Voice...")

        try:
            samples = get_asr_samples(lang, n=SAMPLES_PER_LANG)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Got {len(samples)} samples. Transcribing...\n")

        references = []
        hypotheses = []
        skipped    = 0

        for i, (wav_path, reference) in enumerate(samples, start=1):
            print(f"  [{i:03d}/{len(samples)}] {os.path.basename(wav_path)}")
            print(f"    REF: {reference[:80]}")

            if lang == "zh":
                hyp = transcribe_zh(asr, wav_path)
            else:
                hyp = asr.transcribe(wav_path, language=lang)

            if hyp.startswith("["):
                print(f"    HYP: {hyp} — skipping")
                skipped += 1
                continue

            print(f"    HYP: {hyp[:80]}")
            references.append(reference)
            hypotheses.append(hyp)

        if not hypotheses:
            print(f"  No valid transcriptions for {lang}.")
            continue

        refs_norm = [normalise(r, lang) for r in references]
        hyps_norm = [normalise(h, lang) for h in hypotheses]

        wer       = jiwer.wer(refs_norm, hyps_norm)
        processed = len(hypotheses)
        total     = len(samples)

        print(f"\n  WER: {wer:.4f}  ({wer*100:.1f}%)   "
              f"{processed}/{total} transcribed"
              + (f"  [{skipped} skipped]" if skipped else ""))

        all_results[lang] = {
            "wer": wer, "processed": processed,
            "total": total, "skipped": skipped,
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*58}")
    print("TABLE 1 — ASR Word Error Rate")
    print(f"{'='*58}")
    print(f"{'Language':<18} {'WER':>8}  {'WER %':>8}  {'Processed':>10}")
    print("-"*58)
    for lang, r in all_results.items():
        print(f"{lang.upper():<18} {r['wer']:>8.4f}  {r['wer']*100:>7.1f}%  "
              f"{r['processed']}/{r['total']:>4}")
    print("="*58)
    print("\nNote: Chinese WER uses character-level scoring with OpenCC")
    print("Traditional→Simplified normalization applied before comparison.")
    print(f"Dataset: FLEURS (primary) + Common Voice (supplement), {SAMPLES_PER_LANG} utterances/language.")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("WER RESULTS\n")
        f.write(f"Dataset: FLEURS + Common Voice, {SAMPLES_PER_LANG} utterances per language\n")
        f.write("="*58 + "\n")
        f.write(f"{'Language':<18} {'WER':>8}  {'WER %':>8}  {'Processed':>10}\n")
        f.write("-"*58 + "\n")
        for lang, r in all_results.items():
            f.write(f"{lang.upper():<18} {r['wer']:>8.4f}  {r['wer']*100:>7.1f}%  "
                    f"{r['processed']}/{r['total']:>4}\n")
        f.write("="*58 + "\n")

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()