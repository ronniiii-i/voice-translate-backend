"""
eval_context.py — Table 3: Context vs no-context BLEU comparison

HOW TO RUN:
    From your backend/ folder:
        python evaluation/eval_context.py

WHAT IT DOES:
    Downloads 40 discourse-sensitive sequences per language pair (120 total)
    from OpenSubtitles — consecutive subtitle pairs where the target sentence
    contains a pronoun (he/she/they/it). Each sequence is translated under two
    conditions:

        Condition A: use_context=False  (sentence-level baseline)
        Condition B: use_context=True   (with k=3 conversational history)

    Aggregate BLEU scores under each condition are reported per language pair.

LANGUAGE PAIRS (Table 3):
    EN → FR,  EN → DE,  EN → ZH

DATASET:
    SODA (allenai/soda, CC-BY 4.0) — a million-scale multi-turn conversational
    dialogue dataset from Allen Institute for AI (Kim et al., 2022, EMNLP 2023).
    Each item is a complete, genuinely coherent dialogue where pronouns in later
    turns refer to entities introduced in earlier turns. 40 pronoun-containing
    sequences per language pair = 120 sequences total.

CONTEXT WINDOW:
    k=3 (last 3 utterances), consistent with the paper and mt_model.py.
"""

import os
import sys
import sacrebleu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.mt_model import HelsinkiTranslator
from eval_datasets import get_context_sequences

RESULTS_FILE       = os.path.join(os.path.dirname(__file__), "results_context.txt")
SEQUENCES_PER_PAIR = 40   # 40 × 3 pairs = 120 total

EVAL_PAIRS = [
    ("en", "fr"),
    ("en", "de"),
    ("en", "zh"),
]


def bleu_score(hypothesis: str, reference: str, tgt: str) -> float:
    tokenize = "zh" if tgt == "zh" else "13a"
    return sacrebleu.sentence_bleu(hypothesis, [reference], tokenize=tokenize).score


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
        print(f"Evaluating context effect: {pair_label}")
        print(f"{'='*62}")

        try:
            sequences = get_context_sequences(src, tgt, n=SEQUENCES_PER_PAIR)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            continue

        if not sequences:
            print(f"  No sequences available for {pair_label} — skipping.")
            continue

        print(f"  {len(sequences)} sequences loaded.\n")

        without_bleus: list[float] = []
        with_bleus:    list[float] = []

        for i, seq in enumerate(sequences, start=1):
            context   = seq["context"]
            target    = seq["target"]
            reference = seq["reference"]

            # Condition A: no context
            out_without = translator.translate(target, src=src, tgt=tgt, use_context=False)

            # Condition B: with context window (k=3 applied inside translate())
            out_with = translator.translate(
                target, src=src, tgt=tgt,
                context=context, use_context=True,
            )

            bw = bleu_score(out_without, reference, tgt)
            bc = bleu_score(out_with,    reference, tgt)
            without_bleus.append(bw)
            with_bleus.append(bc)

            line = (
                f"\n  [{i:02d}] Target:   \"{target[:70]}\"\n"
                f"       Context:  {context}\n"
                f"       REF:      \"{reference[:70]}\"\n"
                f"       [A] No context  → \"{out_without[:70]}\"\n"
                f"           BLEU: {bw:.1f}\n"
                f"       [B] With context→ \"{out_with[:70]}\"\n"
                f"           BLEU: {bc:.1f}   Δ={bc-bw:+.1f}"
            )
            print(line)
            output_lines.append(line)

        avg_without = sum(without_bleus) / len(without_bleus)
        avg_with    = sum(with_bleus)    / len(with_bleus)
        improvement = avg_with - avg_without

        summary = (
            f"\n  AGGREGATE {pair_label} (n={len(sequences)}):\n"
            f"    Without context (avg BLEU): {avg_without:.1f}\n"
            f"    With context k=3 (avg BLEU): {avg_with:.1f}\n"
            f"    Improvement:                {improvement:+.1f}"
        )
        print(summary)
        output_lines.append(summary)

        results_by_pair[pair_label] = {
            "without": avg_without,
            "with":    avg_with,
            "delta":   improvement,
            "n":       len(sequences),
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*62}")
    print("TABLE 3 — Sentence-level baseline vs. context window (k=3)")
    print(f"{'='*62}")
    print(f"{'Language Pair':<16} {'Without Context':>17} {'With Context (k=3)':>20} {'Improvement':>13}")
    print("-"*62)
    for pair, r in results_by_pair.items():
        print(f"{pair:<16} {r['without']:>17.1f} {r['with']:>20.1f} {r['delta']:>+13.1f}")
    print("="*62)
    print(f"\nDataset: SODA (allenai/soda, Kim et al. 2022), {SEQUENCES_PER_PAIR} pronoun-containing sequences per pair.")
    print("Context window k=3 applied inside HelsinkiTranslator._resolve_pronouns().")
    print("Note: BLEU computed against source sentence; scores measure translation change,")
    print("not absolute quality. Positive delta = context improved output fluency.")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("CONTEXT A/B EVALUATION RESULTS\n")
        f.write(f"Dataset: OpenSubtitles, {SEQUENCES_PER_PAIR} sequences per language pair\n")
        f.write("="*62 + "\n\n")
        for line in output_lines:
            f.write(line + "\n")
        f.write("\n\nTABLE 3 SUMMARY\n")
        f.write("="*62 + "\n")
        f.write(f"{'Language Pair':<16} {'Without':>10} {'With k=3':>10} {'Delta':>10} {'N':>5}\n")
        f.write("-"*62 + "\n")
        for pair, r in results_by_pair.items():
            f.write(f"{pair:<16} {r['without']:>10.1f} {r['with']:>10.1f} "
                    f"{r['delta']:>+10.1f} {r['n']:>5}\n")
        f.write("="*62 + "\n")

    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()