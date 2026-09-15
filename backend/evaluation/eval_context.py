"""
eval_context.py — Table 3: Context vs no-context BLEU comparison

HOW TO RUN:
    From your backend/ folder:
        python evaluation/eval_context.py

WHAT IT DOES:
    Evaluates context-aware pronoun resolution using the SCAT (Supporting 
    Context for Ambiguous Translations) dataset. It translates 100 ambiguous 
    English-to-French sequences under two conditions:

        Condition A: use_context=False  (sentence-level baseline)
        Condition B: use_context=True   (with conversational history)

    Official corpus-level BLEU scores under each condition are reported.

CONTEXT WINDOW:
    k=3 (last 3 utterances), consistent with the paper and mt_model.py.
"""

import os
import sys
import sacrebleu

# Ensure the root folder is in the python path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.mt_model import HelsinkiTranslator

from eval_datasets import get_scat_sequences 

RESULTS_FILE       = os.path.join(os.path.dirname(__file__), "results_context_scat.txt")
SEQUENCES_PER_PAIR = 100   # Updated to 100 for statistical significance

EVAL_PAIRS = [
    ("en", "fr"),
]

def main():
    print("Initialising HelsinkiTranslator...")
    translator = HelsinkiTranslator()

    for src, tgt in EVAL_PAIRS:
        translator.ensure_pair_loaded(src, tgt)

    results_by_pair: dict[str, dict] = {}
    output_lines:    list[str]       = []

    for src, tgt in EVAL_PAIRS:
        pair_label = f"{src}→{tgt}"
        print(f"\n{'='*62}")
        print(f"Evaluating context effect (SCAT): {pair_label}")
        print(f"{'='*62}")

        try:
            sequences = get_scat_sequences(n=SEQUENCES_PER_PAIR)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            continue

        if not sequences:
            print(f"  No sequences available for {pair_label} — skipping.")
            continue

        print(f"  {len(sequences)} sequences loaded.\n")

        tokenize_method = "13a"

        hypotheses_without = []
        hypotheses_with    = []
        references         = []

        for i, seq in enumerate(sequences, start=1):
            context   = seq["context"]
            target    = seq["target"]
            reference = seq["reference"]

            out_without = translator.translate(target, src=src, tgt=tgt, use_context=False)

            out_with = translator.translate(
                target, src=src, tgt=tgt,
                context=context, use_context=True,
            )

            hypotheses_without.append(out_without)
            hypotheses_with.append(out_with)
            references.append([reference])

            bw = sacrebleu.sentence_bleu(out_without, [reference], tokenize=tokenize_method).score
            bc = sacrebleu.sentence_bleu(out_with, [reference], tokenize=tokenize_method).score

            line = (
                f"\n  [{i:03d}] Target:   \"{target[:70]}\"\n"
                f"       Context:  {context}\n"
                f"       REF:      \"{reference[:70]}\"\n"
                f"       [A] No context  → \"{out_without[:70]}\"\n"
                f"           BLEU: {bw:.1f}\n"
                f"       [B] With context→ \"{out_with[:70]}\"\n"
                f"           BLEU: {bc:.1f}   Δ={bc-bw:+.1f}"
            )
            print(line)
            output_lines.append(line)

        corpus_bleu_without = sacrebleu.corpus_bleu(hypotheses_without, references, tokenize=tokenize_method).score
        corpus_bleu_with    = sacrebleu.corpus_bleu(hypotheses_with, references, tokenize=tokenize_method).score
        corpus_delta        = corpus_bleu_with - corpus_bleu_without

        summary = (
            f"\n  AGGREGATE CORPUS {pair_label} (n={len(sequences)}):\n"
            f"    Without context (Corpus BLEU): {corpus_bleu_without:.1f}\n"
            f"    With context (Corpus BLEU):    {corpus_bleu_with:.1f}\n"
            f"    Improvement (Delta):           {corpus_delta:+.1f}"
        )
        print(summary)
        output_lines.append(summary)

        results_by_pair[pair_label] = {
            "without": corpus_bleu_without,
            "with":    corpus_bleu_with,
            "delta":   corpus_delta,
            "n":       len(sequences),
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*62}")
    print("TABLE 3 — Sentence-level baseline vs. Contextual Resolution (SCAT)")
    print(f"{'='*62}")
    print(f"{'Language Pair':<16} {'Without Context':>17} {'With Context':>20} {'Improvement':>13}")
    print("-"*62)
    for pair, r in results_by_pair.items():
        print(f"{pair:<16} {r['without']:>17.1f} {r['with']:>20.1f} {r['delta']:>+13.1f}")
    print("="*62)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("CONTEXT A/B EVALUATION RESULTS (SCAT)\n")
        f.write(f"Dataset: SCAT (Supporting Context for Ambiguous Translations), {SEQUENCES_PER_PAIR} sequences\n")
        f.write("="*62 + "\n\n")
        for line in output_lines:
            f.write(line + "\n")
        f.write("\n\nTABLE 3 SUMMARY (CORPUS BLEU)\n")
        f.write("="*62 + "\n")
        f.write(f"{'Language Pair':<16} {'Without':>10} {'With':>10} {'Delta':>10} {'N':>5}\n")
        f.write("-"*62 + "\n")
        for pair, r in results_by_pair.items():
            f.write(f"{pair:<16} {r['without']:>10.1f} {r['with']:>10.1f} "
                    f"{r['delta']:>+10.1f} {r['n']:>5}\n")
        f.write("="*62 + "\n")

    print(f"\nFull results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()