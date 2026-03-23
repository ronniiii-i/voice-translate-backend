"""
eval_bleu.py — Table 2: BLEU scores for all 20 directional translation routes

HOW TO RUN:
    From your backend/ folder:
        python evaluation/eval_bleu.py

    If the script was killed mid-run, just rerun it — completed routes are
    saved to results_bleu.txt and skipped automatically on the next run.
"""

import os
import sys
import sacrebleu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.mt_model import HelsinkiTranslator
from sentences import BLEU_SENTENCES

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results_bleu.txt")

DIRECT_ROUTES = [
    ("en", "fr"), ("fr", "en"),
    ("en", "de"), ("de", "en"),
    ("en", "es"), ("es", "en"),
    ("en", "zh"), ("zh", "en"),
    ("fr", "de"), ("de", "fr"),
    ("fr", "es"), ("es", "fr"),
    ("de", "es"), ("es", "de"),
]

PIVOT_ROUTES = [
    ("fr", "zh"), ("zh", "fr"),
    ("de", "zh"), ("zh", "de"),
    ("es", "zh"), ("zh", "es"),
]

# Chinese needs character-level tokenization for meaningful BLEU scores.
# Without this, sacrebleu treats the whole sentence as one token → BLEU = 0.
ZH_ROUTES = {(s, t) for s, t in (DIRECT_ROUTES + PIVOT_ROUTES)
             if s == "zh" or t == "zh"}


def bleu_for_route(src, tgt, hypotheses, references):
    """Compute BLEU with correct tokenization for the language pair."""
    if (src, tgt) in ZH_ROUTES:
        # "zh" tokenizer splits Chinese characters individually
        return sacrebleu.corpus_bleu(hypotheses, [references], tokenize="zh")
    else:
        return sacrebleu.corpus_bleu(hypotheses, [references])


def load_existing_results():
    """Load any results already saved from a previous (interrupted) run."""
    existing = {}
    if not os.path.exists(RESULTS_FILE):
        return existing
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            # Lines look like: "en→fr         73.9  Direct"
            parts = line.split()
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
    route_data  = BLEU_SENTENCES.get((src, tgt))

    if not route_data:
        print(f"  {route_label}: no sentences in sentences.py — skipping")
        return None, route_type

    print(f"\n{route_label}  [{route_type}]")
    translator.ensure_pair_loaded(src, tgt)

    sources    = route_data["sources"]
    references = route_data["references"]
    hypotheses = []

    for i, sentence in enumerate(sources):
        translation = translator.translate(
            sentence, src=src, tgt=tgt, use_context=False,
        )
        hypotheses.append(translation)
        print(f"  [{i+1:02d}] SRC: {sentence}")
        print(f"        SYS: {translation}")
        print(f"        REF: {references[i]}")

    bleu = bleu_for_route(src, tgt, hypotheses, references)
    print(f"  BLEU: {bleu.score:.1f}")
    return bleu.score, route_type


def save_results(all_results):
    with open(RESULTS_FILE, "w") as f:
        f.write("BLEU RESULTS\n")
        f.write("="*55 + "\n")
        f.write(f"{'Route':<12} {'BLEU':>8}  {'Type'}\n")
        f.write("-"*55 + "\n")
        for route, (score, rtype) in all_results.items():
            f.write(f"{route:<12} {score:>8.1f}  {rtype}\n")
        f.write("="*55 + "\n")
        f.write(f"\nSentences per route: 10\n")
        f.write("Note: Chinese routes scored with tokenize='zh' (character-level)\n")


def main():
    print("Initialising HelsinkiTranslator...")
    translator = HelsinkiTranslator()

    # Load any previously completed routes so we can skip them
    existing_scores = load_existing_results()

    # all_results maps route_label → (score, type)
    all_results = {}

    # Restore existing results with placeholder type
    for label, score in existing_scores.items():
        rtype = "Pivot via English" if any(
            label == f"{s}→{t}" for s, t in PIVOT_ROUTES
        ) else "Direct"
        all_results[label] = (score, rtype)

    all_routes = [(r, "Direct") for r in DIRECT_ROUTES] + \
                 [(r, "Pivot via English") for r in PIVOT_ROUTES]

    print("── RUNNING ROUTES ──────────────────────────────────────")

    for (src, tgt), route_type in all_routes:
        label = f"{src}→{tgt}"

        if label in existing_scores:
            print(f"  {label}: already done ({existing_scores[label]:.1f}) — skipping")
            continue

        score, rtype = run_route(translator, src, tgt, route_type)
        if score is not None:
            all_results[label] = (score, rtype)
            # Save after every route so a crash doesn't lose work
            save_results(all_results)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*55}")
    print("TABLE 2 — copy these values into your paper")
    print(f"{'='*55}")
    print(f"{'Route':<12} {'BLEU':>8}  {'Type'}")
    print("-"*55)
    for route, (score, rtype) in all_results.items():
        print(f"{route:<12} {score:>8.1f}  {rtype}")
    print("="*55)
    print("\nNote: Chinese routes use character-level BLEU (tokenize='zh').")
    print("This gives meaningful scores for Chinese unlike the default tokenizer.")

    save_results(all_results)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()