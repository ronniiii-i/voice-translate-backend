"""
eval_wer.py — Table 1: ASR Word Error Rate per language

HOW TO RUN:
    From your backend/ folder:
        python evaluation/eval_wer.py
"""

import os
import sys
import string
import jiwer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.asr_model import WhisperASR
from sentences import ASR_SENTENCES

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
RESULTS_FILE   = os.path.join(os.path.dirname(__file__), "results_wer.txt")

NUMBER_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
    # Chinese number words → digits (covers both scripts)
    "三": "3", "五": "5", "百": "100",
}

# Traditional → Simplified Chinese character mapping for common characters
# that appear in the test sentences. Whisper sometimes outputs Traditional
# when the speaker's accent triggers that variant.
TRAD_TO_SIMP = str.maketrans({
    "報": "报", "告": "告", "發": "发", "給": "给", "嗎": "吗",
    "項": "项", "遲": "迟", "請": "请", "審": "审", "確": "确",
    "們": "们", "個": "个", "與": "与", "戶": "户", "話": "话",
    "運": "运", "經": "经", "業": "业", "問": "问", "題": "题",
    "幫": "帮", "這": "这", "嗎": "吗", "預": "预", "點": "点",
    "貴": "贵", "義": "义", "週": "周", "現": "现", "系": "系",
    "統": "统", "覆": "复", "樓": "楼", "議": "议", "會": "会",
    "間": "间", "訂": "订",
})


def normalise_zh(text: str) -> str:
    """
    Normalise Chinese text for WER scoring:
    1. Convert Traditional → Simplified characters
    2. Insert spaces between characters (jiwer needs space-separated tokens)
    3. Remove punctuation
    4. Normalise digits
    """
    text = text.translate(TRAD_TO_SIMP)
    # Remove punctuation (Chinese and ASCII)
    zh_punct = "。，、？！；：""''（）【】《》…—～·"
    text = text.translate(str.maketrans("", "", zh_punct + string.punctuation))
    # Replace number words
    for word, digit in NUMBER_MAP.items():
        text = text.replace(word, digit)
    # Insert space between every character so jiwer can tokenise
    text = " ".join(list(text.strip()))
    return text


def normalise(text: str, lang: str = "") -> str:
    """Normalise text before WER scoring."""
    if lang == "zh":
        return normalise_zh(text)
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [NUMBER_MAP.get(w, w) for w in text.split()]
    return " ".join(words)


def transcribe_zh(asr, filepath):
    """
    Transcribe Chinese audio with an initial prompt that biases Whisper
    toward Simplified Chinese output rather than Traditional.
    """
    import os
    if not os.path.exists(filepath):
        return "[file_not_found]"
    if os.path.getsize(filepath) < 1000:
        return "[file_too_small]"

    import time
    t0 = time.time()
    try:
        segments, _ = asr.model.transcribe(
            filepath,
            language="zh",
            initial_prompt="以下是普通话的转录。",  # "The following is a Mandarin transcription."
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
        print(f"[ASR] {time.time()-t0:.1f}s → '{text}'")
        return text if text else "[silence]"
    except Exception as e:
        print(f"[ASR] Exception: {e}")
        return "[error]"


def main():
    print("Loading WhisperASR (base model)...")
    asr = WhisperASR(model_size="base")
    print("Model loaded.\n")

    all_results = {}

    for lang, ground_truths in ASR_SENTENCES.items():
        print(f"── Language: {lang} ──────────────────────────")
        references = []
        hypotheses = []
        skipped    = 0

        for i, gt in enumerate(ground_truths, start=1):
            filename = f"{lang}_{i:02d}.wav"
            filepath = os.path.join(RECORDINGS_DIR, filename)

            if not os.path.exists(filepath):
                print(f"  MISSING: {filename} — skipping")
                skipped += 1
                continue

            # Use specialised Chinese transcription to force Simplified output
            if lang == "zh":
                transcription = transcribe_zh(asr, filepath)
            else:
                transcription = asr.transcribe(filepath, language=lang)

            if transcription.startswith("["):
                print(f"  {filename}: ASR returned '{transcription}' — skipping")
                skipped += 1
                continue

            references.append(gt)
            hypotheses.append(transcription)
            print(f"  {filename}")
            print(f"    REF: {gt}")
            print(f"    HYP: {transcription}")

        if not hypotheses:
            print(f"  No valid results for {lang} — check recordings.\n")
            continue

        refs_norm = [normalise(r, lang) for r in references]
        hyps_norm = [normalise(h, lang) for h in hypotheses]

        wer = jiwer.wer(refs_norm, hyps_norm)

        processed = len(hypotheses)
        total     = len(ground_truths)
        print(f"\n  WER: {wer:.4f}  ({wer*100:.1f}%)   "
              f"{processed}/{total} files processed"
              + (f"  [{skipped} skipped]" if skipped else ""))
        print()
        all_results[lang] = {
            "wer": wer, "processed": processed,
            "total": total, "skipped": skipped,
        }

    print("\n" + "="*55)
    print("SUMMARY — copy these values into Table 1 of your paper")
    print("="*55)
    print(f"{'Language':<12} {'WER':>8}  {'WER %':>8}  {'Files':>10}")
    print("-"*55)
    for lang, r in all_results.items():
        print(f"{lang:<12} {r['wer']:>8.4f}  {r['wer']*100:>7.1f}%  "
              f"{r['processed']}/{r['total']:>4}")
    print("="*55)
    print("\nNote: Chinese WER uses character-level scoring with")
    print("Traditional→Simplified normalisation applied before comparison.")

    with open(RESULTS_FILE, "w") as f:
        f.write("WER RESULTS\n")
        f.write("="*55 + "\n")
        f.write(f"{'Language':<12} {'WER':>8}  {'WER %':>8}  {'Files':>10}\n")
        f.write("-"*55 + "\n")
        for lang, r in all_results.items():
            f.write(f"{lang:<12} {r['wer']:>8.4f}  {r['wer']*100:>7.1f}%  "
                    f"{r['processed']}/{r['total']:>4}\n")
        f.write("="*55 + "\n")

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()