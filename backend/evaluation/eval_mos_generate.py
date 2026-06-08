"""
eval_mos_generate.py — Generate TTS audio clips for the MOS survey (Table 4)

HOW TO RUN:
    From your backend/ folder:
        python evaluation/eval_mos_generate.py

WHAT IT DOES:
    Synthesises 5 audio clips per language = 25 WAV files total.
    Saves them to: evaluation/mos_clips/
    Overwrites any previous clips — this is a fresh generation.

SURVEY DESIGN (split-listener to avoid fatigue):
    Divide your listener pool into 5 groups of ≥20 people each.
    Each group rates only one language (5 clips) on a 1–5 ITU-T P.800 scale.
    This gives per-language MOS estimates without fatiguing any individual listener.

    Column format expected by eval_mos_calculate.py:
        "Clip EN 01 — Click to listen, then rate"
        "Clip EN 02 — Click to listen, then rate"
        ...
        "Clip ZH 05 — Click to listen, then rate"

AFTER RUNNING THIS SCRIPT:
    1. Upload mos_clips/ to Google Drive (make each file shareable)
    2. Create 5 Google Forms (one per language) with audio links and 1–5 rating
    3. Distribute each form to its assigned listener group
    4. Export all responses as a single merged CSV
    5. Run eval_mos_calculate.py with the merged CSV as mos_responses.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.tts_model import PiperTTS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "mos_clips")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5 sentences per language — varied topics, length, and phonetic patterns.
# Sentences were selected to represent a range of phonetic complexity and
# natural conversational speech patterns across different registers.
TTS_SENTENCES = {
    "en": [
        "Have you heard about the new café that opened downtown last Saturday?",
        "She laughed at the thought of running a marathon in this heat.",
        "I need three things from the store: milk, bread, and something sweet.",
        "The children were playing outside when the storm suddenly began.",
        "Would you mind turning down the volume just a little?",
    ],
    "fr": [
        "Avez-vous entendu parler du nouveau café qui a ouvert samedi dernier?",
        "Elle a ri à l'idée de courir un marathon par cette chaleur.",
        "J'ai besoin de trois choses au magasin: du lait, du pain et quelque chose de sucré.",
        "Les enfants jouaient dehors quand l'orage a soudainement éclaté.",
        "Pourriez-vous baisser le volume un peu s'il vous plaît?",
    ],
    "de": [
        "Haben Sie von dem neuen Café gehört, das letzten Samstag in der Innenstadt eröffnet hat?",
        "Sie lachte bei dem Gedanken, bei dieser Hitze einen Marathon zu laufen.",
        "Ich brauche drei Dinge aus dem Laden: Milch, Brot und etwas Süßes.",
        "Die Kinder spielten draußen, als das Gewitter plötzlich begann.",
        "Würden Sie die Lautstärke bitte ein bisschen leiser stellen?",
    ],
    "es": [
        "¿Has escuchado hablar del nuevo café que abrió el sábado pasado en el centro?",
        "Ella se rió al pensar en correr un maratón con este calor.",
        "Necesito tres cosas de la tienda: leche, pan y algo dulce.",
        "Los niños jugaban afuera cuando la tormenta comenzó de repente.",
        "¿Te importaría bajar el volumen un poco por favor?",
    ],
    "zh": [
        "你听说上周六市中心新开的那家咖啡馆了吗？",
        "她一想到在这种炎热天气里跑马拉松就忍不住笑了。",
        "我需要从商店买三样东西：牛奶、面包和一些甜食。",
        "孩子们在外面玩耍，暴风雨突然开始了。",
        "能请你把音量调低一点吗？",
    ],
}


def main():
    print("Initialising PiperTTS...")
    tts = PiperTTS()
    print("TTS ready.\n")

    generated = []

    for lang, sentences in TTS_SENTENCES.items():
        print(f"── {lang.upper()} ──────────────────────────────────────────")
        for i, sentence in enumerate(sentences, start=1):
            filename = f"mos_{lang}_{i:02d}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)

            try:
                tts.synthesize(sentence, filepath, language=lang)
                size_kb = os.path.getsize(filepath) / 1024
                print(f"  {filename} ({size_kb:.0f} KB) — \"{sentence[:60]}\"")
                generated.append((filename, sentence))
            except Exception as e:
                print(f"  ERROR generating {filename}: {e}")

    print(f"\nGenerated {len(generated)}/25 clips in {OUTPUT_DIR}/")

    # Write manifest
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("MOS CLIP MANIFEST\n")
        f.write("="*60 + "\n")
        f.write("Survey design: 5 listener groups, 1 language each, ≥20 listeners/group\n")
        f.write("Rating scale: 1 (very unnatural) – 5 (fully natural), ITU-T P.800\n")
        f.write("="*60 + "\n\n")
        current_lang = None
        for filename, sentence in generated:
            lang = filename.split("_")[1]
            if lang != current_lang:
                f.write(f"\n[{lang.upper()}] — Group {list(TTS_SENTENCES.keys()).index(lang)+1}\n")
                current_lang = lang
            f.write(f"  {filename}: {sentence}\n")

    print(f"Manifest saved to {manifest_path}")
    print("\nNext steps:")
    print("  1. Upload mos_clips/ contents to Google Drive")
    print("  2. Create 5 Google Forms (one per language), add audio links + 1–5 rating")
    print("  3. Target ≥20 listeners per form")
    print("  4. Merge all responses into mos_responses.csv")
    print("  5. Run: python evaluation/eval_mos_calculate.py")


if __name__ == "__main__":
    main()
