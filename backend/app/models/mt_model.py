import re
import threading
import time
from transformers import MarianMTModel, MarianTokenizer

# ── Language pair → HuggingFace model name ───────────────────────────────────
MODEL_MAP: dict[tuple[str, str], str] = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("fr", "de"): "Helsinki-NLP/opus-mt-fr-de",
    ("de", "fr"): "Helsinki-NLP/opus-mt-de-fr",
    ("fr", "es"): "Helsinki-NLP/opus-mt-fr-es",
    ("es", "fr"): "Helsinki-NLP/opus-mt-es-fr",
    ("de", "es"): "Helsinki-NLP/opus-mt-de-es",
    ("es", "de"): "Helsinki-NLP/opus-mt-es-de",
}

PIVOT_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("fr", "zh"), ("zh", "fr"),
    ("de", "zh"), ("zh", "de"),
    ("es", "zh"), ("zh", "es"),
})

CONTEXT_WINDOW = 3

_PRONOUN_SUBJECT = re.compile(
    r'\b(he|she|they|it|him|her|them|his|hers|their|its)\b',
    re.IGNORECASE
)

_NAME_PATTERN = re.compile(r'\b[A-Z][a-z]{2,}\b')


def _resolve_pronouns(text: str, context: list[str]) -> str:
    if not _PRONOUN_SUBJECT.search(text):
        return text  

    candidate = None
    for utt in reversed(context):
        names = _NAME_PATTERN.findall(utt)
        names = [n for n in names if n not in (
            "The", "A", "An", "This", "That", "These", "Those",
            "I", "You", "We", "They", "He", "She", "It"
        )]
        if names:
            candidate = names[-1]
            break

    if not candidate:
        return text 

    resolved = _PRONOUN_SUBJECT.sub(candidate, text)
    if resolved != text:
        print(f"[MT] Pronoun resolved: '{text}' → '{resolved}' "
              f"(candidate: {candidate})")
    return resolved


class HelsinkiTranslator:
    def __init__(self):
        self._models: dict[tuple[str, str], MarianMTModel] = {}
        self._tokenizers: dict[tuple[str, str], MarianTokenizer] = {}
        self._load_locks: dict[tuple[str, str], threading.Lock] = {}
        self._global_lock = threading.Lock()
        print("✅ HelsinkiTranslator ready (lazy-load mode)")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_load_lock(self, pair: tuple[str, str]) -> threading.Lock:
        with self._global_lock:
            if pair not in self._load_locks:
                self._load_locks[pair] = threading.Lock()
            return self._load_locks[pair]

    def _load_pair(self, src: str, tgt: str) -> None:
        pair = (src, tgt)
        lock = self._get_load_lock(pair)
        with lock:
            if pair in self._models:
                return
            model_name = MODEL_MAP.get(pair)
            if model_name is None:
                raise ValueError(
                    f"[MT] No direct model for {src}→{tgt}."
                )
            print(f"[MT] Loading {src}→{tgt} ({model_name})...")
            t0 = time.time()
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model.eval()
            self._tokenizers[pair] = tokenizer
            self._models[pair] = model
            print(f"[MT] ✅ {src}→{tgt} ready ({time.time() - t0:.1f}s)")

    def _translate_direct(self, text: str, src: str, tgt: str) -> str:
        import torch
        pair = (src, tgt)
        tokenizer = self._tokenizers[pair]
        model = self._models[pair]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def ensure_pair_loaded(self, src: str, tgt: str) -> None:
        pair = (src, tgt)
        if pair in MODEL_MAP:
            if pair not in self._models:
                self._load_pair(src, tgt)
        elif pair in PIVOT_PAIRS:
            if (src, "en") not in self._models:
                self._load_pair(src, "en")
            if ("en", tgt) not in self._models:
                self._load_pair("en", tgt)
        else:
            print(f"[MT] ⚠️  No route defined for {src}→{tgt}")

    def translate(
        self,
        text: str,
        src: str,
        tgt: str,
        context: list[str] | None = None,
        use_context: bool = True,
    ) -> str:
        if src == tgt:
            return text

        # Apply pronoun resolution heuristic (English source only for now)
        input_text = text
        if use_context and context and src == "en":
            recent = context[-CONTEXT_WINDOW:]
            input_text = _resolve_pronouns(text, recent)

        pair = (src, tgt)

        # ── Direct translation ────────────────────────────────────────────────
        if pair in MODEL_MAP:
            if pair not in self._models:
                self._load_pair(src, tgt)
            result = self._translate_direct(input_text, src, tgt)
            print(f"[MT] context={'on' if use_context and context else 'off'} "
                  f"input='{input_text[:60]}' output='{result[:60]}'")
            return result

        # ── Pivot through English ─────────────────────────────────────────────
        if pair in PIVOT_PAIRS:
            en_pair  = (src, "en")
            tgt_pair = ("en", tgt)
            if en_pair  not in self._models: self._load_pair(src, "en")
            if tgt_pair not in self._models: self._load_pair("en", tgt)
            en_text = self._translate_direct(input_text, src, "en")
            result  = self._translate_direct(en_text, "en", tgt)
            print(f"[MT] pivot {src}→en→{tgt}: '{result[:60]}'")
            return result

        print(f"[MT] ⚠️  No route for {src}→{tgt}, returning original")
        return text