"""
eval_datasets.py — Shared dataset loading for all evaluation scripts.

Downloads evaluation data from HuggingFace Hub using streaming mode
(no full-corpus downloads). Audio is cached locally to avoid re-downloading
between runs.

Datasets used:
  ASR:     google/fleurs (primary) + facebook/voxpopuli (supplement, EN/FR/DE/ES only)
           Note: Mozilla Common Voice was removed from HuggingFace in October 2025.
  BLEU:    open_subtitles (parallel sentence pairs, conversational)
  Context: open_subtitles (consecutive subtitle pairs filtered for pronouns)

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

# OpenSubtitles uses ISO 639-1; Chinese is "zh" in their index
OPENSUBS_LANG = {
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


def get_bleu_pairs(src: str, tgt: str, n: int = 23, seed: int = 42) -> list[tuple[str, str]]:
    """
    Returns list of (source_sentence, reference_translation) for BLEU evaluation.

    Source: OpenSubtitles parallel corpus — conversational, subtitle-domain text.
    Sentences are filtered to MIN_WORDS–MAX_WORDS range for quality.
    """
    cache_name = f"bleu_{src}_{tgt}_{n}_{seed}"
    cached = _load_cache(cache_name)
    if cached:
        print(f"[cache] BLEU {src}→{tgt}: {len(cached)} pairs")
        return cached

    from datasets import load_dataset

    l1   = OPENSUBS_LANG.get(src, src)
    l2   = OPENSUBS_LANG.get(tgt, tgt)
    rng  = random.Random(seed)
    pairs: list[tuple[str, str]] = []

    # Try (l1, l2) then (l2, l1) — OpenSubtitles requires alphabetical ordering
    for a, b, swapped in [(l1, l2, False), (l2, l1, True)]:
        try:
            print(f"[datasets] OpenSubtitles lang1={a} lang2={b}...")
            ds = load_dataset(
                "opus_open_subtitles", lang1=a, lang2=b,
                split="train", streaming=True,
            )
            pool: list[tuple[str, str]] = []
            for item in ds:
                tr = item.get("translation", {})
                s  = tr.get(a, "").strip()
                t  = tr.get(b, "").strip()
                if not s or not t:
                    continue
                if MIN_WORDS <= _wc(s) <= MAX_WORDS and MIN_WORDS <= _wc(t) <= MAX_WORDS:
                    pool.append((t, s) if swapped else (s, t))
                if len(pool) >= n * 10:
                    break

            if pool:
                rng.shuffle(pool)
                pairs = pool[:n]
                print(f"  → {len(pairs)} pairs")
                break

        except Exception as e:
            print(f"  OpenSubtitles {a}-{b} failed: {e}")

    if not pairs:
        raise RuntimeError(
            f"Could not obtain BLEU pairs for {src}→{tgt}. "
            "Check network connectivity and OpenSubtitles availability."
        )

    _save_cache(cache_name, pairs)
    return pairs


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

    l1  = OPENSUBS_LANG.get(src, src)
    l2  = OPENSUBS_LANG.get(tgt, tgt)
    rng = random.Random(seed)
    seqs: list[dict] = []

    # Only try (l1=src, l2=tgt) — context eval always has src=en
    for a, b, swapped in [(l1, l2, False), (l2, l1, True)]:
        try:
            print(f"[datasets] OpenSubtitles {a}-{b} for context sequences...")
            ds = load_dataset(
                "opus_open_subtitles", lang1=a, lang2=b,
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
            print(f"  OpenSubtitles {a}-{b} failed: {e}")

    if not seqs:
        raise RuntimeError(
            f"Could not obtain context sequences for {src}→{tgt}. "
            "Check network connectivity and OpenSubtitles availability."
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
