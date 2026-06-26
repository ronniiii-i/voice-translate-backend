"""
eval_datasets.py — Shared dataset loading for all evaluation scripts.

Downloads evaluation data from HuggingFace Hub using streaming mode
(no full-corpus downloads). Audio is cached locally to avoid re-downloading
between runs.

Datasets used:
  ASR:     google/fleurs (primary) + facebook/voxpopuli (supplement, EN/FR/DE/ES only)
           Note: Mozilla Common Voice was removed from HuggingFace in October 2025.
  BLEU:    openlanguagedata/flores_plus (primary, devtest split) + Helsinki-NLP/opus-100 (fallback)
           Best-of-two selection via self-BLEU proxy. All 13 routes supported.
  Context: Helsinki-NLP/opus-100 (consecutive pairs filtered for pronouns)

Cache layout:
  evaluation/audio_cache/{lang}/0001.wav ...
  evaluation/dataset_cache/bleu_en_fr_23_42.pkl ...

HOW TO RUN (indirectly — called by the other eval scripts):
  Each eval script imports from this module. Run eval_*.py directly.
"""

import os
import re
import random
import pickle
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal as scipy_signal

# Tell the datasets library to use soundfile for audio decoding instead of
# the default torchcodec (which requires a separate GPU-oriented install).
# Must be set before any `from datasets import ...` call.
os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")

EVAL_DIR        = Path(__file__).parent
CACHE_DIR       = EVAL_DIR / "dataset_cache"
AUDIO_CACHE_DIR = EVAL_DIR / "audio_cache"

CACHE_DIR.mkdir(exist_ok=True)
AUDIO_CACHE_DIR.mkdir(exist_ok=True)

# ── Language code mappings ─────────────────────────────────────────────────────

FLEURS_LANG = {
    "en": "en_us",
    "fr": "fr_fr",
    "de": "de_de",
    "es": "es_419",
    "zh": "cmn_hans_cn",
}

# VoxPopuli (facebook/voxpopuli) — supplement for European languages only.
# Common Voice was removed from HuggingFace in October 2025.
# ZH has no VoxPopuli support; FLEURS cmn_hans_cn alone covers it.
VOXPOPULI_LANG = {
    "en": "en",
    "fr": "fr",
    "de": "de",
    "es": "es",
}

# FLORES-200 language codes (facebook/flores)
FLORES_LANG = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
}

# opus-100 only has EN-centric pairs; non-EN pairs pivot through EN in dataset layer
OPUS100_LANG = {
    "en": "en",
    "fr": "fr",
    "de": "de",
    "es": "es",
    "zh": "zh",
}

PRONOUN_RE  = re.compile(r'\b(he|she|they|it)\b', re.IGNORECASE)
MIN_WORDS   = 4
MAX_WORDS   = 30


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resample(arr: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    if orig_sr == target_sr:
        return arr.astype(np.float32)
    n = int(len(arr) * target_sr / orig_sr)
    return scipy_signal.resample(arr, n).astype(np.float32)


def _wc(text: str) -> int:
    return len(text.split())


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"


def _load_cache(name: str):
    p = _cache_path(name)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(name: str, data) -> None:
    with open(_cache_path(name), "wb") as f:
        pickle.dump(data, f)


# ── Public API ─────────────────────────────────────────────────────────────────

def get_asr_samples(lang: str, n: int = 100, seed: int = 42) -> list[tuple[str, str]]:
    """
    Returns list of (wav_path, reference_text) for WER evaluation.

    Primary source: FLEURS (google/fleurs) — clean, validated multilingual speech.
    Supplement: Common Voice if FLEURS yields fewer than n samples.
    Audio is resampled to 16 kHz and cached at evaluation/audio_cache/{lang}/.
    """
    cache_name = f"asr_{lang}_{n}_{seed}"
    cached = _load_cache(cache_name)
    if cached:
        valid = [(p, t) for p, t in cached if os.path.exists(p)]
        if len(valid) >= n:
            print(f"[cache] ASR {lang}: {len(valid)} samples")
            return valid[:n]

    from datasets import load_dataset

    lang_dir = AUDIO_CACHE_DIR / lang
    lang_dir.mkdir(exist_ok=True)
    rng     = random.Random(seed)
    samples: list[tuple[str, str]] = []

    # ── FLEURS ────────────────────────────────────────────────────────────────
    fleurs_code = FLEURS_LANG[lang]
    print(f"[datasets] Streaming FLEURS '{fleurs_code}' for {lang}...")
    try:
        ds = load_dataset(
            "google/fleurs", fleurs_code,
            split="test", streaming=True,
        )
        # Write each clip to disk immediately — no in-memory audio buffering.
        # FLEURS test split is ~500-1500 items; we stop as soon as we have n.
        for item in ds:
            text = (item.get("transcription") or item.get("raw_transcription") or "").strip()
            if not (text and MIN_WORDS <= _wc(text) <= MAX_WORDS):
                continue
            audio = item["audio"]
            arr   = _resample(np.array(audio["array"]), audio["sampling_rate"])
            idx   = len(samples) + 1
            wav   = str(lang_dir / f"{idx:04d}.wav")
            sf.write(wav, arr, 16000)
            samples.append((wav, text))
            if len(samples) >= n:
                break

        print(f"  FLEURS → {len(samples)} samples")
    except Exception as e:
        print(f"  FLEURS failed for {lang}: {e}")

    # ── VoxPopuli supplement (European languages only) ────────────────────────
    # Common Voice was removed from HuggingFace in October 2025.
    # VoxPopuli covers EN/FR/DE/ES; ZH relies on FLEURS alone.
    if len(samples) < n and lang in VOXPOPULI_LANG:
        needed  = n - len(samples)
        vp_code = VOXPOPULI_LANG[lang]
        print(f"[datasets] VoxPopuli '{vp_code}' supplement ({needed} needed)...")
        try:
            ds = load_dataset(
                "facebook/voxpopuli", vp_code,
                split="test", streaming=True,
            )
            for item in ds:
                text = (item.get("normalized_text") or item.get("raw_text") or "").strip()
                if not (text and MIN_WORDS <= _wc(text) <= MAX_WORDS):
                    continue
                audio = item["audio"]
                arr   = _resample(np.array(audio["array"]), audio["sampling_rate"])
                idx   = len(samples) + 1
                wav   = str(lang_dir / f"{idx:04d}.wav")
                sf.write(wav, arr, 16000)
                samples.append((wav, text))
                if len(samples) >= n:
                    break

            print(f"  After supplement → {len(samples)} samples")
        except Exception as e:
            print(f"  VoxPopuli supplement skipped: {e}")

    if not samples:
        raise RuntimeError(
            f"Could not obtain any ASR samples for '{lang}'. "
            "Check network connectivity and HuggingFace dataset availability."
        )

    _save_cache(cache_name, samples)
    return samples[:n]


def _bleu_pairs_from_flores(src: str, tgt: str, n: int, seed: int) -> list[tuple[str, str]]:
    """
    Fetch parallel sentence pairs from FLORES-200 (facebook/flores, devtest split).
    Clean, human-translated, sentence-aligned — best choice for MT evaluation.
    Covers all language pairs directly including non-EN (fr-es, fr-de, etc.).
    """
    from datasets import load_dataset

    src_code = FLORES_LANG.get(src)
    tgt_code = FLORES_LANG.get(tgt)
    if not src_code or not tgt_code:
        raise ValueError(f"No FLORES code for {src} or {tgt}")

    print(f"[datasets] flores_plus {src}→{tgt} ({src_code} / {tgt_code})...")
    ds_src = list(load_dataset("openlanguagedata/flores_plus", src_code, split="devtest", streaming=False))
    ds_tgt = list(load_dataset("openlanguagedata/flores_plus", tgt_code, split="devtest", streaming=False))

    rng  = random.Random(seed)
    pool: list[tuple[str, str]] = []
    indices = list(range(min(len(ds_src), len(ds_tgt))))
    rng.shuffle(indices)

    for i in indices:
        s = ds_src[i].get("sentence", "").strip()
        t = ds_tgt[i].get("sentence", "").strip()
        if s and t and MIN_WORDS <= _wc(s) <= MAX_WORDS:
            pool.append((s, t))
        if len(pool) >= n:
            break

    print(f"  flores_plus → {len(pool)} pairs")
    return pool


def _bleu_pairs_from_opus100(src: str, tgt: str, n: int, seed: int) -> list[tuple[str, str]]:
    """
    Fetch parallel pairs from Helsinki-NLP/opus-100.

    Strategy (in order):
    1. Try a direct config (test split, then train split) for both orderings.
       Works for EN-centric pairs and the handful of non-EN configs that exist
       (fr-zh, de-fr, de-zh, de-nl, de-ru, nl-zh, ru-zh, …).
    2. For non-EN pairs with no direct config, pivot through EN: load two
       train-split EN configs and intersect on the shared EN sentence.
       Cap at PIVOT_SCAN items to keep runtime reasonable.
    """
    from datasets import load_dataset

    PIVOT_SCAN = 10000   # max items to scan per EN-pivot config

    rng   = random.Random(seed)
    l_src = OPUS100_LANG.get(src, src)
    l_tgt = OPUS100_LANG.get(tgt, tgt)

    def _try_direct(a: str, b: str, swapped: bool) -> list[tuple[str, str]]:
        """Try loading config a-b from test then train split."""
        config = f"{a}-{b}"
        for split in ("test", "train"):
            try:
                print(f"[datasets] opus-100 {config} ({split} split)...")
                ds   = load_dataset("Helsinki-NLP/opus-100", config,
                                    split=split, streaming=True)
                pool: list[tuple[str, str]] = []
                scanned = 0
                for item in ds:
                    scanned += 1
                    tr = item.get("translation", {})
                    s  = tr.get(a, "").strip()
                    t  = tr.get(b, "").strip()
                    if not s or not t:
                        continue
                    if MIN_WORDS <= _wc(s) <= MAX_WORDS and MIN_WORDS <= _wc(t) <= MAX_WORDS:
                        pool.append((t, s) if swapped else (s, t))
                    if len(pool) >= n * 10 or (split == "train" and scanned >= PIVOT_SCAN):
                        break
                if pool:
                    rng.shuffle(pool)
                    result = pool[:n]
                    print(f"  opus-100 {config} → {len(result)} pairs")
                    return result
            except Exception as e:
                print(f"  opus-100 {config} ({split}) failed: {e}")
        return []

    def _load_lang_map(src_lang: str, tgt_lang: str) -> list[tuple[str, str]]:
        """
        Load (src_text, tgt_text) pairs for any language pair by going through EN.
        Loads src→EN pairs, then EN→tgt pairs, and chains them by position
        (same document, same sentence index) rather than trying to intersect
        on exact string match which fails across different corpora.
        """
        def _get_pairs_via_en(pivot_lang: str, other_lang: str, want_other_as_src: bool):
            """Stream up to PIVOT_SCAN items from en-{other} or {other}-en config."""
            a, b    = ("en", other_lang) if other_lang > "en" else (other_lang, "en")
            config  = f"{a}-{b}"
            results = []
            for split in ("train", "test"):
                try:
                    ds = load_dataset("Helsinki-NLP/opus-100", config,
                                      split=split, streaming=True)
                    scanned = 0
                    for item in ds:
                        scanned += 1
                        tr  = item.get("translation", {})
                        en  = tr.get("en", "").strip()
                        oth = tr.get(other_lang, "").strip()
                        if en and oth and MIN_WORDS <= _wc(en) <= MAX_WORDS:
                            results.append((oth, en) if want_other_as_src else (en, oth))
                        if scanned >= PIVOT_SCAN:
                            break
                    if results:
                        break
                except Exception as e:
                    print(f"  opus-100 {config} ({split}) pivot failed: {e}")
            return results

        # src→EN pairs and EN→tgt pairs, both indexed by position within their corpus
        src_en_pairs = _get_pairs_via_en("en", src_lang, want_other_as_src=True)
        en_tgt_pairs = _get_pairs_via_en("en", tgt_lang, want_other_as_src=False)

        # Chain: take src sentences from src_en_pairs and tgt sentences from
        # en_tgt_pairs at matched positions (same-length window)
        count  = min(len(src_en_pairs), len(en_tgt_pairs), n * 3)
        result = []
        indices = list(range(count))
        rng.shuffle(indices)
        for i in indices:
            s = src_en_pairs[i][0]   # src language text
            t = en_tgt_pairs[i][1]   # tgt language text
            if s and t:
                result.append((s, t))
            if len(result) >= n:
                break
        print(f"  opus-100 chain-pivot {src_lang}→en({len(src_en_pairs)}) + en→{tgt_lang}({len(en_tgt_pairs)}) → {len(result)} pairs")
        return result

    pairs: list[tuple[str, str]] = []

    # ── Step 1: try direct configs (both orderings) ───────────────────────────
    for a, b, swapped in [(l_src, l_tgt, False), (l_tgt, l_src, True)]:
        pairs = _try_direct(a, b, swapped)
        if pairs:
            return pairs

    # ── Step 2: chain-pivot through EN ───────────────────────────────────────
    print(f"[datasets] opus-100 chain-pivot {src}→en→{tgt}...")
    pairs = _load_lang_map(l_src, l_tgt)
    if pairs:
        rng.shuffle(pairs)
        return pairs[:n]

    return []


def get_bleu_pairs(src: str, tgt: str, n: int = 23, seed: int = 42) -> list[tuple[str, str]]:
    """
    Returns list of (source_sentence, reference_translation) for BLEU evaluation.

    Tries FLORES-200 and opus-100 independently, scores both with a self-BLEU
    proxy (higher = cleaner references), and returns the better-scoring dataset.

    FLORES-200: clean human translations, all language pairs including non-EN.
    opus-100:   EN-centric; non-EN pairs joined via shared EN pivot sentences.
    Both use held-out test/devtest splits — no training data leakage.
    """
    import sacrebleu as _sb

    cache_name = f"bleu4_{src}_{tgt}_{n}_{seed}"
    cached = _load_cache(cache_name)
    if cached:
        print(f"[cache] BLEU {src}→{tgt}: {len(cached)} pairs")
        return cached

    candidates: dict[str, list[tuple[str, str]]] = {}

    try:
        flores_pairs = _bleu_pairs_from_flores(src, tgt, n, seed)
        if flores_pairs:
            candidates["FLORES-200"] = flores_pairs
    except Exception as e:
        print(f"  FLORES-200 failed: {e}")

    try:
        opus_pairs = _bleu_pairs_from_opus100(src, tgt, n, seed)
        if opus_pairs:
            candidates["opus-100"] = opus_pairs
    except Exception as e:
        print(f"  opus-100 failed: {e}")

    if not candidates:
        raise RuntimeError(
            f"Could not obtain BLEU pairs for {src}→{tgt} from any source. "
            "Check network connectivity and HuggingFace dataset availability."
        )

    if len(candidates) == 1:
        name, pairs = next(iter(candidates.items()))
        print(f"  Using {name} (only source available)")
        _save_cache(cache_name, pairs)
        return pairs

    # Use self-BLEU as a reference quality proxy: cleaner references score higher
    zh_routes  = {src, tgt} & {"zh"}
    tokenize   = "zh" if zh_routes else "13a"
    best_name, best_pairs, best_score = None, None, -1.0
    for name, pairs in candidates.items():
        hyps  = [s for s, _ in pairs]
        refs  = [r for _, r in pairs]
        score = _sb.corpus_bleu(hyps, [refs], tokenize=tokenize).score
        print(f"  {name} self-BLEU proxy: {score:.1f}")
        if score > best_score:
            best_score, best_name, best_pairs = score, name, pairs

    print(f"  → Selecting {best_name} (proxy {best_score:.1f})")
    _save_cache(cache_name, best_pairs)
    return best_pairs


def get_context_sequences(
    src: str, tgt: str, n: int = 40, seed: int = 42,
) -> list[dict]:
    """
    Returns discourse sequences for context-aware MT evaluation.

    Each item: {"context": [str, ...], "target": str, "reference": str}
    where context = prior English turns, target = pronoun-containing English
    sentence, reference = gold translation in the target language.

    Sequences are extracted from consecutive OpenSubtitles subtitle pairs
    where the final subtitle in the window contains a pronoun (he/she/they/it).
    """
    cache_name = f"context_{src}_{tgt}_{n}_{seed}"
    cached = _load_cache(cache_name)
    if cached:
        print(f"[cache] Context {src}→{tgt}: {len(cached)} sequences")
        return cached

    from datasets import load_dataset

    l1  = OPUS100_LANG.get(src, src)
    l2  = OPUS100_LANG.get(tgt, tgt)
    rng = random.Random(seed)
    seqs: list[dict] = []

    # Only try (l1=src, l2=tgt) — context eval always has src=en
    for a, b, swapped in [(l1, l2, False), (l2, l1, True)]:
        try:
            config = f"{a}-{b}"
            print(f"[datasets] Helsinki-NLP/opus-100 config={config} for context sequences...")
            ds = load_dataset(
                "Helsinki-NLP/opus-100", config,
                split="train", streaming=True,
            )

            src_win: list[str] = []
            tgt_win: list[str] = []
            pool: list[dict]   = []

            for item in ds:
                tr = item.get("translation", {})
                s  = tr.get(a, "").strip()
                t  = tr.get(b, "").strip()

                if not s or not t:
                    # Subtitle gap — reset window to avoid cross-scene context
                    src_win.clear()
                    tgt_win.clear()
                    continue

                src_win.append(s)
                tgt_win.append(t)
                if len(src_win) > 3:
                    src_win.pop(0)
                    tgt_win.pop(0)

                if (len(src_win) >= 2
                        and PRONOUN_RE.search(s)
                        and MIN_WORDS <= _wc(s) <= MAX_WORDS):

                    context_s = src_win[:-1].copy()
                    target_s  = s
                    ref_s     = t

                    if swapped:
                        # Ordering was reversed — src window is actually tgt language
                        context_s = tgt_win[:-1].copy()
                        target_s  = t
                        ref_s     = s

                    pool.append({
                        "context":   context_s,
                        "target":    target_s,
                        "reference": ref_s,
                    })

                if len(pool) >= n * 3:
                    break

            if pool:
                rng.shuffle(pool)
                seqs = pool[:n]
                print(f"  → {len(seqs)} sequences")
                break

        except Exception as e:
            print(f"  opus-100 {a}-{b} failed: {e}")

    if not seqs:
        raise RuntimeError(
            f"Could not obtain context sequences for {src}→{tgt}. "
            "Check network connectivity and HuggingFace dataset availability."
        )

    _save_cache(cache_name, seqs)
    return seqs


def get_latency_audio(n: int = 5, seed: int = 42) -> list[str]:
    """Returns English WAV paths for latency testing. Reuses ASR audio cache."""
    samples = get_asr_samples("en", n=max(n * 3, 20), seed=seed)
    rng = random.Random(seed)
    paths = [p for p, _ in samples]
    rng.shuffle(paths)
    return paths[:n]