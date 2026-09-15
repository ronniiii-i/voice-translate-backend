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

# Only replace SUBJECT pronouns — never object pronouns (him, her, it, them, its)
# _PRONOUN_SUBJECT_ONLY = re.compile(
#     r'\b(he|she|they)\b',
#     re.IGNORECASE
# )

# _NAME_PATTERN = re.compile(r'\b[A-Z][a-z]{2,}\b')

# _NOT_NAMES = {
#     # Articles / determiners
#     "The", "A", "An", "This", "That", "These", "Those", "Some", "Any",
#     "All", "Both", "Each", "Every", "Neither", "Either",
#     # Pronouns
#     "I", "You", "We", "They", "He", "She", "It", "Who", "What",
#     "Which", "Whose", "Whom",
#     # Auxiliary / modal verbs
#     "Have", "Has", "Had", "Can", "Could", "Would", "Should",
#     "Will", "Shall", "May", "Might", "Must", "Do", "Does", "Did",
#     "Be", "Is", "Are", "Was", "Were", "Been", "Being",
#     # Common sentence starters that get capitalised
#     "Please", "Thank", "Yes", "No", "Not", "But", "And", "Or", "So",
#     "Well", "Oh", "Okay", "Now", "Then", "Here", "There", "Just",
#     "Let", "Get", "Got", "Put", "See", "Say", "Said", "Tell", "Told",
#     "Come", "Go", "Make", "Take", "Give", "Know", "Think", "Look",
#     "Want", "Need", "Try", "Keep", "Use", "Find", "From", "With",
#     "About", "After", "Before", "Because", "When", "Where", "How",
#     "If", "As", "At", "By", "In", "On", "Of", "To", "For", "Up",
#     "Down", "Out", "Off", "Over", "Into", "Also", "Only", "Even",
#     "Still", "Already", "Never", "Always", "Maybe", "Really",
#     # Titles (abbreviated — the full name follows)
#     "Mr", "Mrs", "Ms", "Dr", "Sir", "Lord", "Lady", "Captain",
#     "General", "President", "Professor",
#     # Common words falsely capitalised mid-sentence in subtitles
#     "Don", "Are", "Did", "Does", "Has", "Was", "Were", "His", "Her",
#     "Its", "Our", "Your", "Their", "My", "We", "Us", "Him", "Them",
#     "Rule", "Note", "Dear", "Hello", "Hey", "Hi", "Bye", "Yeah",
#     "Okay", "Right", "Sure", "Sorry", "Wait", "Stop", "Help",
#     "Good", "Bad", "New", "Old", "Big", "Small", "Long", "Short",
#     "First", "Last", "Next", "Other", "Same", "Different",
#     "Very", "Too", "More", "Most", "Less", "Few", "Many", "Much",
#     # Legal / document words common in opus-100
#     "Article", "Section", "Chapter", "Part", "Annex", "Pursuant",
#     "Whereas", "Having", "According", "Provided", "Subject",
#     "Economic", "Social", "National", "International", "European",
#     "United", "Federal", "General", "Special", "Official",
#     # Misc corpus noise
#     "Laureate", "Necessity", "Health", "Team", "Example",
# }

# _SUBJECT_NAME_PATTERN = re.compile(
#     r'(?:^|[.!?]\s+|[-–]\s*)([A-Z][a-z]{2,})(?:\s+(?:is|was|has|had|will|would|can|could|did|does|said|says|told|asked|went|came|got|took|made|looked|seemed|wanted|needed|tried|smiled|laughed|nodded|shook|turned|walked|ran|stood|sat|lay)\b)',
# )


# def _resolve_pronouns(text: str, context: list[str]) -> str:
#     if not _PRONOUN_SUBJECT_ONLY.search(text):
#         return text

#     candidate = None

#     for utt in reversed(context):
#         matches = _SUBJECT_NAME_PATTERN.findall(utt)
#         valid = [m for m in matches if m not in _NOT_NAMES and len(m) >= 3]
#         if valid:
#             candidate = valid[-1]
#             break

#     if not candidate:
#         for utt in reversed(context):
#             names = _NAME_PATTERN.findall(utt)
#             valid = [n for n in names if n not in _NOT_NAMES and len(n) >= 4]
#             if valid:
#                 candidate = valid[-1]
#                 break

#     if not candidate:
#         return text

#     resolved = _PRONOUN_SUBJECT_ONLY.sub(candidate, text)
#     if resolved != text:
#         print(f"[MT] Pronoun resolved: '{text}' → '{resolved}' "
#               f"(candidate: {candidate})")
#     return resolved

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

        pair = (src, tgt)

        if pair in MODEL_MAP:
            if pair not in self._models:
                self._load_pair(src, tgt)
        elif pair in PIVOT_PAIRS:
            if (src, "en") not in self._models: self._load_pair(src, "en")
            if ("en", tgt) not in self._models: self._load_pair("en", tgt)
        else:
            print(f"[MT] ⚠️ No route for {src}→{tgt}, returning original")
            return text

        def _run_inference(input_str: str) -> str:
            if pair in MODEL_MAP:
                return self._translate_direct(input_str, src, tgt)
            elif pair in PIVOT_PAIRS:
                en_text = self._translate_direct(input_str, src, "en")
                return self._translate_direct(en_text, "en", tgt)
            return input_str

        standalone_translation = _run_inference(text).strip()

        if not use_context or not context:
            return standalone_translation

        recent_context = context[-CONTEXT_WINDOW:]
        context_str = " ".join(recent_context).strip()

        if not context_str:
            return standalone_translation

        combined_text = f"{context_str}\n{text}"

        translated_combined = _run_inference(combined_text).strip()

        if '\n' in translated_combined:
            final_output = translated_combined.split('\n')[-1].strip()
            
            if final_output:
                return final_output
        return standalone_translation