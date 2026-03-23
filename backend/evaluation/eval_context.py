"""
eval_context.py — Table 3: Context vs no-context BLEU comparison

HOW TO RUN:
    From your backend/ folder:
        python evaluation/eval_context.py

WHAT IT DOES:
    1. Takes the 8 discourse-dependent sequences from sentences.py
    2. For each sequence, translates the final turn TWO ways:
         Condition A: use_context=False  (sentence-level baseline)
         Condition B: use_context=True   (with prior turn as context)
    3. Compares both against a reference translation using BLEU
    4. Prints the comparison and saves to evaluation/results_context.txt

WHY THIS MATTERS:
    This is the core contribution of your system. The results here go into
    Table 3 of your paper. Even a small but consistent improvement in
    Condition B over A is sufficient to demonstrate the value of the
    context window, consistent with Voita et al. (2018).

EXPECTED OUTPUT (example):
    Sequence 1: "She said the project is delayed."
      Without context → "Elle a dit que le projet est retardé."     BLEU: 48.2
      With context    → "Maria a dit que le projet est en retard."   BLEU: 61.7
      Improvement: +13.5

    ...
    AGGREGATE en→fr:  without=51.3  with=62.8  improvement=+11.5
"""

import os
import sys
import sacrebleu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.mt_model import HelsinkiTranslator
from sentences import CONTEXT_SEQUENCES

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results_context.txt")

# Which target languages to evaluate context effect on
# (en→fr, en→de, en→zh are the three in Table 3)
EVAL_PAIRS = [
    ("en", "fr", "reference_fr"),
    ("en", "de", "reference_de"),
    ("en", "zh", "reference_zh"),
]


def main():
    print("Initialising HelsinkiTranslator...")
    translator = HelsinkiTranslator()

    # Pre-load all needed models
    for src, tgt, _ in EVAL_PAIRS:
        translator.ensure_pair_loaded(src, tgt)

    results_by_pair = {}
    output_lines    = []

    for src, tgt, ref_key in EVAL_PAIRS:
        pair_label = f"{src}→{tgt}"
        print(f"\n{'='*60}")
        print(f"Evaluating context effect: {pair_label}")
        print(f"{'='*60}")

        without_bleus = []
        with_bleus    = []

        for seq in CONTEXT_SEQUENCES:
            target    = seq["target"]
            context   = seq["context"]
            reference = seq[ref_key]
            note      = seq.get("note", "")

            # Condition A: no context
            out_without = translator.translate(
                target,
                src=src,
                tgt=tgt,
                use_context=False,
            )

            # Condition B: with context
            out_with = translator.translate(
                target,
                src=src,
                tgt=tgt,
                context=context,
                use_context=True,
            )

            # Sentence-level BLEU for each (multiply by 100 for readability)
            bleu_without = sacrebleu.sentence_bleu(out_without, [reference], tokenize=("zh" if tgt == "zh" else "13a")).score
            bleu_with    = sacrebleu.sentence_bleu(out_with,    [reference], tokenize=("zh" if tgt == "zh" else "13a")).score

            without_bleus.append(bleu_without)
            with_bleus.append(bleu_with)

            line = (
                f"\n  Target:  \"{target}\"\n"
                f"  Note:    {note}\n"
                f"  Context: {context}\n"
                f"  REF:     \"{reference}\"\n"
                f"  [A] Without context → \"{out_without}\"\n"
                f"      BLEU: {bleu_without:.1f}\n"
                f"  [B] With context    → \"{out_with}\"\n"
                f"      BLEU: {bleu_with:.1f}\n"
                f"  Δ BLEU: {bleu_with - bleu_without:+.1f}"
            )
            print(line)
            output_lines.append(line)

        avg_without = sum(without_bleus) / len(without_bleus)
        avg_with    = sum(with_bleus)    / len(with_bleus)
        improvement = avg_with - avg_without

        summary = (
            f"\n  AGGREGATE {pair_label}:\n"
            f"    Without context (avg BLEU): {avg_without:.1f}\n"
            f"    With context    (avg BLEU): {avg_with:.1f}\n"
            f"    Improvement:               {improvement:+.1f}"
        )
        print(summary)
        output_lines.append(summary)

        results_by_pair[pair_label] = {
            "without": avg_without,
            "with":    avg_with,
            "delta":   improvement,
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("TABLE 3 — copy these values into your paper")
    print(f"{'='*60}")
    print(f"{'Language Pair':<16} {'Without Context':>17} {'With Context (k=3)':>20} {'Improvement':>13}")
    print("-"*60)
    for pair, r in results_by_pair.items():
        print(f"{pair:<16} {r['without']:>17.1f} {r['with']:>20.1f} {r['delta']:>+13.1f}")
    print("="*60)
    print("\nNote: improvement is most meaningful on pronoun/anaphora sequences.")
    print("Even a small consistent positive delta supports the context contribution claim.")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("CONTEXT A/B EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        for line in output_lines:
            f.write(line + "\n")
        f.write("\n\nTABLE 3 SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"{'Language Pair':<16} {'Without':>10} {'With':>10} {'Delta':>10}\n")
        f.write("-"*60 + "\n")
        for pair, r in results_by_pair.items():
            f.write(f"{pair:<16} {r['without']:>10.1f} {r['with']:>10.1f} {r['delta']:>+10.1f}\n")
        f.write("="*60 + "\n")

    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()