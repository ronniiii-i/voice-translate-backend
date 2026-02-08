import argostranslate.translate
import time

print("🚀 Loading models (this is the only slow part)...")
start_load = time.time()
# Warm up the engine
argostranslate.translate.translate("Warm up", "en", "fr")
print(f"✅ Loaded in {time.time() - start_load:.2f}s")

sentences = [
    "Hello, how are you?",
    "Building an AI app is fun.",
    "The translation should be fast now.",
    "Testing the i5 processor performance."
]

for s in sentences:
    start_tx = time.time()
    result = argostranslate.translate.translate(s, "en", "fr")
    print(f"⏱️ {time.time() - start_tx:.4f}s | {result}")