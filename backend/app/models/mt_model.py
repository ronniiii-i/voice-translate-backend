import threading
import time
from transformers import MarianMTModel, MarianTokenizer

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

class HelsinkiTranslator:
    def __init__(self):
        self._models: dict[tuple[str, str], MarianMTModel] = {}
        self._tokenizers: dict[tuple[str, str], MarianTokenizer] = {}
        self._load_locks: dict[tuple[str, str], threading.Lock] = {}
        self._global_lock = threading.Lock()
        print("✅ HelsinkiTranslator ready (lazy-load mode)")


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
                raise ValueError(f"[MT] No direct model for {src}→{tgt}. "
                                 f"Use ensure_pair_loaded for pivot pairs.")

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

    def ensure_pair_loaded(self, src: str, tgt: str) -> None:
        pair = (src, tgt)

        if pair in MODEL_MAP:
            if pair not in self._models:
                self._load_pair(src, tgt)
        elif pair in PIVOT_PAIRS:
            if (src, "en") in MODEL_MAP and (src, "en") not in self._models:
                self._load_pair(src, "en")
            if ("en", tgt) in MODEL_MAP and ("en", tgt) not in self._models:
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
        
        if use_context and context:
            recent = context[-CONTEXT_WINDOW:]
            input_text = " | ".join(recent) + " | " + text
        else:
            input_text = text

        pair = (src, tgt)

        if pair in MODEL_MAP:
            if pair not in self._models:
                self._load_pair(src, tgt)
            return self._translate_direct(input_text, src, tgt)

        if pair in PIVOT_PAIRS:
            en_pair = (src, "en")
            if en_pair not in self._models:
                self._load_pair(src, "en")
            en_text = self._translate_direct(input_text, src, "en")

            tgt_pair = ("en", tgt)
            if tgt_pair not in self._models:
                self._load_pair("en", tgt)
            return self._translate_direct(en_text, "en", tgt)

        print(f"[MT] ⚠️  No route for {src}→{tgt}, returning original")
        return text