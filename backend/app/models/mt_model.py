import argostranslate.translate

class ArgosTranslator:
    def __init__(self, warmup_pairs=None):
        """
        Initializes the translator and warms up specified language pairs.
        warmup_pairs: List of tuples like [("en", "fr"), ("ko", "zh")]
        """
        print("Initializing MT Engine...")
        
        if warmup_pairs:
            for src, tgt in warmup_pairs:
                print(f"⏳ Priming {src} -> {tgt}...")
                try:
                    argostranslate.translate.translate("1", src, tgt)
                except Exception as e:
                    print(f"⚠️ Could not warm up {src}-{tgt}: {e}")
        
        print("✅ MT Engine Ready.")
        
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between languages"""
        return argostranslate.translate.translate(text, source_lang, target_lang)
