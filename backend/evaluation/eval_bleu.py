"""
eval_bleu.py — Table 2: BLEU scores for 13 directional translation routes

HOW TO RUN:
    From your backend/ folder:
        python evaluation/eval_bleu.py

    Completed routes are saved after each one — safe to interrupt and resume.

WHAT IT DOES:
    Downloads ~23 parallel sentence pairs per route from OpenSubtitles
    (conversational, subtitle-domain text). Translates each source sentence
    using HelsinkiTranslator and scores against corpus references with sacrebleu.
    Routes without a direct Helsinki-NLP model pivot through English automatically.

ROUTES EVALUATED (13 total matching Table 2):
    Direct (10): EN↔FR, EN↔DE, EN↔ES, EN↔ZH, FR↔ES, FR↔DE
    Pivot  (3):  FR→ZH, DE→ZH, ES→ZH

DATASET:
    OpenSubtitles parallel corpus via HuggingFace (conversational text).
    ~23 sentences per route ≈ 300 total sentence pairs.
"""

import os
import sys
import sacrebleu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.mt_model import HelsinkiTranslator
from eval_datasets import get_bleu_pairs

RESULTS_FILE     = os.path.join(os.path.dirname(__file__), "results_bleu.txt")
SENTENCES_PER_ROUTE = 23   # ~300 total across 13 routes

DIRECT_ROUTES = [
    ("en", "fr"), ("fr", "en"),
    ("en", "de"), ("de", "en"),
    ("en", "es"), ("es", "en"),
    ("en", "zh"), ("zh", "en"),
    ("fr", "es"),
    ("fr", "de"),
]

PIVOT_ROUTES = [
    ("fr", "zh"),
    ("de", "zh"),
    ("es", "zh"),
]

ZH_ROUTES = {(s, t) for s, t in (DIRECT_ROUTES + PIVOT_ROUTES)
             if s == "zh" or t == "zh"}


def bleu_for_route(src, tgt, hypotheses, references):
    if (src, tgt) in ZH_ROUTES:
        return sacrebleu.corpus_bleu(hypotheses, [references], tokenize="zh")
    return sacrebleu.corpus_bleu(hypotheses, [references])


def load_existing_results():
    existing = {}
    if not os.path.exists(RESULTS_FILE):
        return existing
    with open(RESULTS_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and "→" in parts[0]:
                try:
                    existing[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    if existing:
        print(f"Resuming — {len(existing)} route(s) already done: "
              f"{', '.join(existing.keys())}\n")
    return existing


def run_route(translator, src, tgt, route_type):
    route_label = f"{src}→{tgt}"
    print(f"\n{route_label}  [{route_type}]")

    try:
        pairs = get_bleu_pairs(src, tgt, n=SENTENCES_PER_ROUTE)
    except RuntimeError as e:
        print(f"  ERROR loading pairs: {e}")
        return None, route_type

    if not pairs:
        print(f"  No pairs available — skipping")
        return None, route_type

    translator.ensure_pair_loaded(src, tgt)

    sources    = [s for s, _ in pairs]
    references = [r for _, r in pairs]
    hypotheses = []

    for i, (sentence, reference) in enumerate(zip(sources, references)):
        translation = translator.translate(sentence, src=src, tgt=tgt, use_context=False)
        hypotheses.append(translation)
        print(f"  [{i+1:02d}] SRC: {sentence[:70]}")
        print(f"        SYS: {translation[:70]}")
        print(f"        REF: {reference[:70]}")

    bleu = bleu_for_route(src, tgt, hypotheses, references)
    print(f"  BLEU: {bleu.score:.1f}  (n={len(pairs)})")
    return bleu.score, route_type


def save_results(all_results):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("BLEU RESULTS\n")
        f.write(f"Dataset: OpenSubtitles, {SENTENCES_PER_ROUTE} sentences per route\n")
        f.write("="*58 + "\n")
        f.write(f"{'Route':<12} {'BLEU':>8}  {'Type'}\n")
        f.write("-"*58 + "\n")
        for route, (score, rtype) in all_results.items():
            f.write(f"{route:<12} {score:>8.1f}  {rtype}\n")
        f.write("="*58 + "\n")
        f.write("Note: Chinese routes scored with tokenize='zh' (character-level)\n")
        f.write("Pivot routes translate via English automatically.\n")


def main():
    print("Initialising HelsinkiTranslator...")
    translator = HelsinkiTranslator()

    existing_scores = load_existing_results()

    all_results = {}
    for label, score in existing_scores.items():
        rtype = "Pivot via English" if any(
            label == f"{s}→{t}" for s, t in PIVOT_ROUTES
        ) else "Direct"
        all_results[label] = (score, rtype)

    all_routes = (
        [(r, "Direct")           for r in DIRECT_ROUTES] +
        [(r, "Pivot via English") for r in PIVOT_ROUTES]
    )

    print("── RUNNING ROUTES ───────────────────────────────────────────")

    for (src, tgt), route_type in all_routes:
        label = f"{src}→{tgt}"
        if label in existing_scores:
            print(f"  {label}: already done ({existing_scores[label]:.1f}) — skipping")
            continue

        score, rtype = run_route(translator, src, tgt, route_type)
        if score is not None:
            all_results[label] = (score, rtype)
            save_results(all_results)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*58}")
    print("TABLE 2 — BLEU scores per directional translation route")
    print(f"{'='*58}")
    print(f"{'Route':<12} {'BLEU':>8}  {'Type'}")
    print("-"*58)
    for route, (score, rtype) in all_results.items():
        print(f"{route:<12} {score:>8.1f}  {rtype}")
    print("="*58)
    print(f"\n{len(all_results)} routes evaluated, {SENTENCES_PER_ROUTE} sentences each.")
    print("Chinese routes use character-level BLEU (tokenize='zh').")
    print("Pivot routes compound two Helsinki-NLP models via English.")

    save_results(all_results)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
