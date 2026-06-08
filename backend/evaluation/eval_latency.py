"""
eval_latency.py — Table 5: End-to-end pipeline latency per language pair

HOW TO RUN:
    From your backend/ folder:
        python evaluation/eval_latency.py

    Run eval_wer.py at least once first so the English audio cache is populated.
    If the cache is empty, this script will download English FLEURS samples automatically.

WHAT IT DOES:
    Times the full pipeline (ASR + MT + TTS) for 5 English source WAV files
    per language pair. Reports avg latency and min/max range per pair.
    Saves results to evaluation/results_latency.txt

NOTE ON LATENCY DEFINITION:
    Pipeline latency only — measured from WAV file input to synthesised audio
    output on the server. WebSocket network round-trip is excluded as it varies
    by connection. State clearly in the paper:
    "Pipeline latency was measured as the interval from WAV input to synthesised
    audio output on the server, excluding network transmission time."

LANGUAGE PAIRS (8 categories matching Table 5):
    Direct: EN/FR, EN/DE, EN/ES, EN/ZH, FR/DE
    Pivot:  FR/ZH, DE/ZH, ES/ZH
"""

import os
import sys
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.translation_pipeline import TranslationPipeline
from eval_datasets import get_latency_audio

RESULTS_FILE    = os.path.join(os.path.dirname(__file__), "results_latency.txt")
SAMPLES_PER_PAIR = 5

TEST_PAIRS = [
    ("en", "fr", "EN/FR (direct)"),
    ("en", "de", "EN/DE (direct)"),
    ("en", "es", "EN/ES (direct)"),
    ("en", "zh", "EN/ZH (direct)"),
    ("fr", "de", "FR/DE (direct)"),
    ("fr", "zh", "FR/ZH (pivot)"),
    ("de", "zh", "DE/ZH (pivot)"),
    ("es", "zh", "ES/ZH (pivot)"),
]


def time_pipeline(pipeline, wav_path: str, src: str, tgt: str) -> float | None:
    t0 = time.time()
    try:
        result = pipeline.process_audio(wav_path, source_lang=src, target_lang=tgt, context=None)
        elapsed = time.time() - t0

        if result is None:
            return None
        out_path, src_text, _ = result

        if out_path and os.path.exists(out_path):
            os.unlink(out_path)

        if src_text and src_text.startswith("["):
            return None

        return elapsed
    except Exception as e:
        print(f"    Error: {e}")
        return None


def dominant_stage(src: str, tgt: str, label: str) -> str:
    if "pivot" in label.lower():
        return "ASR + MT ×2"
    if tgt == "zh":
        return "ASR + TTS"
    return "ASR"


def main():
    print("Initialising Translation Pipeline (loads ASR model)...")
    pipeline = TranslationPipeline()
    print("Pipeline ready.\n")

    print(f"Fetching {SAMPLES_PER_PAIR} English source WAV files...")
    source_wavs = get_latency_audio(n=SAMPLES_PER_PAIR)

    if not source_wavs:
        print("ERROR: Could not obtain source WAV files.")
        print("Run eval_wer.py first to populate the audio cache, or check connectivity.")
        return

    print(f"Using {len(source_wavs)} source WAV files per language pair.\n")

    all_results = {}

    for src, tgt, label in TEST_PAIRS:
        print(f"── {label} ─────────────────────────────────────────────")
        pipeline.translator.ensure_pair_loaded(src, tgt)

        timings = []
        for wav in source_wavs:
            elapsed = time_pipeline(pipeline, wav, src, tgt)
            if elapsed is not None:
                timings.append(elapsed)
                print(f"  {os.path.basename(wav)} → {elapsed:.2f}s")
            else:
                print(f"  {os.path.basename(wav)} → skipped (sentinel/error)")

        if not timings:
            print(f"  No valid timings for {label}\n")
            continue

        avg   = statistics.mean(timings)
        mn    = min(timings)
        mx    = max(timings)
        stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
        dom   = dominant_stage(src, tgt, label)

        print(f"\n  Avg: {avg:.1f}s  Min: {mn:.1f}s  Max: {mx:.1f}s  "
              f"StDev: {stdev:.2f}s  Dominant: {dom}\n")

        all_results[label] = {
            "avg": avg, "min": mn, "max": mx,
            "stdev": stdev, "dominant": dom, "n": len(timings),
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("TABLE 5 — End-to-end pipeline latency per language pair")
    print(f"{'='*72}")
    print(f"{'Language Pair':<24} {'Avg (s)':>8} {'Range (s)':>16} {'Dominant Stage'}")
    print("-"*72)
    for label, r in all_results.items():
        rng = f"{r['min']:.1f} – {r['max']:.1f}"
        print(f"{label:<24} {r['avg']:>8.1f} {rng:>16} {r['dominant']}")
    print("="*72)
    print(f"\nSamples per pair: {SAMPLES_PER_PAIR}. Pipeline latency only (excludes network).")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("LATENCY RESULTS\n")
        f.write(f"Samples per pair: {SAMPLES_PER_PAIR}  |  Pipeline latency only\n")
        f.write("="*72 + "\n")
        f.write(f"{'Language Pair':<24} {'Avg':>7} {'Min':>7} {'Max':>7} "
                f"{'StDev':>7} {'N':>4}  Dominant Stage\n")
        f.write("-"*72 + "\n")
        for label, r in all_results.items():
            f.write(f"{label:<24} {r['avg']:>7.2f} {r['min']:>7.2f} {r['max']:>7.2f} "
                    f"{r['stdev']:>7.2f} {r['n']:>4}  {r['dominant']}\n")
        f.write("="*72 + "\n")

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
