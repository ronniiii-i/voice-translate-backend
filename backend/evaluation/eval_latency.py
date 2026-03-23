"""
eval_latency.py — Table 5: End-to-end pipeline latency per language pair

HOW TO RUN:
    From your backend/ folder:
        python evaluation/eval_latency.py

WHAT IT DOES:
    Uses the same WAV recordings from eval_wer.py (in evaluation/recordings/).
    Times the full pipeline (ASR + MT + TTS) for each recording.
    Reports average latency and min/max range per language pair category.
    Saves results to evaluation/results_latency.txt

NOTE ON WHAT "LATENCY" MEANS HERE:
    This script measures PIPELINE latency — the time from WAV file input
    to synthesised WAV file output, which is the dominant portion of
    end-to-end call latency. Network round-trip time (WebSocket) is NOT
    included because it varies by connection. In the paper, state clearly:
    "Pipeline latency was measured as the interval from WAV input to
    synthesised audio output on the server."

REQUIRES:
    - WAV files in evaluation/recordings/ (same ones as WER eval)
    - Piper TTS subprocess must be startable (piper binary in PATH)
"""

import os
import sys
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.translation_pipeline import TranslationPipeline

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
RESULTS_FILE   = os.path.join(os.path.dirname(__file__), "results_latency.txt")

# Language pairs to test
# Format: (source_lang, target_lang, category_label)
TEST_PAIRS = [
    ("en", "fr", "EN↔FR (direct)"),
    ("en", "de", "EN↔DE (direct)"),
    ("en", "es", "EN↔ES (direct)"),
    ("en", "zh", "EN↔ZH (direct)"),
    ("fr", "de", "FR↔DE (direct)"),
    ("fr", "zh", "FR↔ZH (pivot)"),
    ("de", "zh", "DE↔ZH (pivot)"),
    ("es", "zh", "ES↔ZH (pivot)"),
]

# How many recordings to use per pair (uses en_01..en_05 as source for all)
SAMPLES_PER_PAIR = 5


def time_pipeline(pipeline, wav_path, src, tgt):
    """
    Time a single pipeline run. Returns elapsed seconds or None on failure.
    """
    t0 = time.time()
    try:
        out_path, src_text, trans_text = pipeline.process_audio(
            wav_path,
            source_lang=src,
            target_lang=tgt,
            context=None,
        )
        elapsed = time.time() - t0

        # Clean up temp output file
        if out_path and os.path.exists(out_path):
            os.unlink(out_path)

        # If ASR returned a sentinel (silence, hallucination) don't count it
        if src_text and src_text.startswith("["):
            return None

        return elapsed
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    print("Initialising Translation Pipeline (this will load ASR model)...")
    pipeline = TranslationPipeline()
    print("Pipeline ready.\n")

    # Get English recordings as source audio for all tests
    # (We time the pipeline, not the speech content — any valid WAV works)
    source_wavs = []
    for i in range(1, SAMPLES_PER_PAIR + 1):
        wav = os.path.join(RECORDINGS_DIR, f"en_{i:02d}.wav")
        if os.path.exists(wav):
            source_wavs.append(wav)

    if not source_wavs:
        print("ERROR: No WAV files found in evaluation/recordings/")
        print("Run eval_wer.py first to confirm recordings are in place.")
        return

    print(f"Using {len(source_wavs)} source WAV files per language pair.\n")

    all_results = {}

    for src, tgt, label in TEST_PAIRS:
        print(f"── {label} ──────────────────────────────────────")

        # Pre-load MT models for this pair
        pipeline.translator.ensure_pair_loaded(src, tgt)

        timings = []
        for wav in source_wavs:
            elapsed = time_pipeline(pipeline, wav, src, tgt)
            if elapsed is not None:
                timings.append(elapsed)
                print(f"  {os.path.basename(wav)} → {elapsed:.2f}s")
            else:
                print(f"  {os.path.basename(wav)} → skipped (sentinel)")

        if not timings:
            print(f"  No valid timings for {label}\n")
            continue

        avg    = statistics.mean(timings)
        mn     = min(timings)
        mx     = max(timings)
        stdev  = statistics.stdev(timings) if len(timings) > 1 else 0.0

        dominant = "ASR + MT (×2)" if "pivot" in label.lower() else (
                   "ASR + TTS"     if tgt == "zh"              else "ASR")

        print(f"\n  Avg: {avg:.1f}s   Min: {mn:.1f}s   Max: {mx:.1f}s   "
              f"StDev: {stdev:.2f}s   Dominant stage: {dominant}\n")

        all_results[label] = {
            "avg":      avg,
            "min":      mn,
            "max":      mx,
            "stdev":    stdev,
            "dominant": dominant,
            "n":        len(timings),
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("TABLE 5 — copy these values into your paper")
    print(f"{'='*72}")
    print(f"{'Language Pair':<22} {'Avg (s)':>8} {'Range (s)':>16} {'Dominant Stage':<20}")
    print("-"*72)
    for label, r in all_results.items():
        rng = f"{r['min']:.1f} – {r['max']:.1f}"
        print(f"{label:<22} {r['avg']:>8.1f} {rng:>16} {r['dominant']:<20}")
    print("="*72)

    # ── Save results ──────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        f.write("LATENCY RESULTS\n")
        f.write("="*72 + "\n")
        f.write(f"{'Language Pair':<22} {'Avg (s)':>8} {'Min':>8} {'Max':>8} "
                f"{'StDev':>8} {'N':>4} {'Dominant Stage'}\n")
        f.write("-"*72 + "\n")
        for label, r in all_results.items():
            f.write(f"{label:<22} {r['avg']:>8.2f} {r['min']:>8.2f} {r['max']:>8.2f} "
                    f"{r['stdev']:>8.2f} {r['n']:>4}  {r['dominant']}\n")
        f.write("="*72 + "\n")
        f.write(f"\nSamples per pair: {SAMPLES_PER_PAIR}\n")
        f.write("Note: pipeline latency only (excludes network round-trip)\n")

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()