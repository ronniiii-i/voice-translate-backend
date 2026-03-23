"""
eval_mos_calculate.py — Compute MOS mean and standard deviation from survey responses

HOW TO RUN:
    From your backend/ folder:
        python evaluation/eval_mos_calculate.py

REQUIRES:
    evaluation/mos_responses.csv
    (exported from Google Sheets after your survey is complete)

WHAT IT DOES:
    Reads all ratings from your Google Forms CSV export.
    Groups ratings by language (EN, FR, DE, ES, ZH).
    Computes mean and standard deviation per language.
    Prints Table 4 values and saves to evaluation/results_mos.txt
"""

import os
import sys
import csv
import statistics

RESPONSES_FILE = os.path.join(os.path.dirname(__file__), "mos_responses.csv")
RESULTS_FILE   = os.path.join(os.path.dirname(__file__), "results_mos.txt")

# Maps language code to the prefix in your Google Form question titles
# e.g. "Clip EN 01" starts with "Clip EN"
LANG_PREFIXES = {
    "en": "Clip EN",
    "fr": "Clip FR",
    "de": "Clip DE",
    "es": "Clip ES",
    "zh": "Clip ZH",
}

VOICE_MODELS = {
    "en": "en_US-ryan-low",
    "fr": "fr_FR-siwis-low",
    "de": "de_DE-thorsten-low",
    "es": "es_ES-mls_10246-low",
    "zh": "zh_CN-huayan-x_low",
}

QUALITY_LEVELS = {
    "en": "Low",
    "fr": "Low",
    "de": "Low",
    "es": "Low",
    "zh": "X-Low",
}


def main():
    if not os.path.exists(RESPONSES_FILE):
        print(f"ERROR: {RESPONSES_FILE} not found.")
        print("Export your Google Forms responses as CSV and save to that path.")
        print("See MOS_SURVEY_INSTRUCTIONS.txt for how to do this.")
        return

    # Read CSV
    with open(RESPONSES_FILE, newline="", encoding="utf-8") as f:
        reader  = csv.DictReader(f)
        headers = reader.fieldnames
        rows    = list(reader)

    print(f"Loaded {len(rows)} responses with {len(headers)} columns.\n")

    # Group column names by language
    lang_columns = {lang: [] for lang in LANG_PREFIXES}
    for header in headers:
        for lang, prefix in LANG_PREFIXES.items():
            if prefix.lower() in header.lower():
                lang_columns[lang].append(header)

    print("Column mapping:")
    for lang, cols in lang_columns.items():
        print(f"  {lang}: {len(cols)} columns found")
    print()

    # Collect all ratings per language
    lang_ratings = {lang: [] for lang in LANG_PREFIXES}

    for row in rows:
        for lang, cols in lang_columns.items():
            for col in cols:
                val = row.get(col, "").strip()
                try:
                    rating = int(val)
                    if 1 <= rating <= 5:
                        lang_ratings[lang].append(rating)
                    else:
                        print(f"  Warning: out-of-range rating {rating} in column '{col}'")
                except ValueError:
                    if val:  # non-empty but non-numeric
                        print(f"  Warning: non-numeric value '{val}' in column '{col}'")

    # Compute statistics
    results = {}
    for lang, ratings in lang_ratings.items():
        if not ratings:
            print(f"  No ratings found for {lang} — check column names in your CSV")
            continue
        mean = statistics.mean(ratings)
        std  = statistics.stdev(ratings) if len(ratings) > 1 else 0.0
        results[lang] = {
            "mean":    mean,
            "std":     std,
            "n":       len(ratings),
            "model":   VOICE_MODELS[lang],
            "quality": QUALITY_LEVELS[lang],
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TABLE 4 — copy these values into your paper")
    print(f"{'='*70}")
    print(f"{'Language':<12} {'Voice Model':<26} {'Quality':<10} {'MOS (Mean ± SD)'}")
    print("-"*70)
    for lang, r in results.items():
        mos_str = f"{r['mean']:.1f} ± {r['std']:.1f}"
        print(f"{lang.upper():<12} {r['model']:<26} {r['quality']:<10} {mos_str}  (n={r['n']})")
    print("="*70)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        f.write("MOS RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"{'Language':<12} {'Voice Model':<26} {'Quality':<10} {'Mean':>6} {'SD':>6} {'N':>4}\n")
        f.write("-"*70 + "\n")
        for lang, r in results.items():
            f.write(f"{lang.upper():<12} {r['model']:<26} {r['quality']:<10} "
                    f"{r['mean']:>6.2f} {r['std']:>6.2f} {r['n']:>4}\n")
        f.write("="*70 + "\n")
        f.write(f"\nTotal responses: {len(rows)}\n")

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()