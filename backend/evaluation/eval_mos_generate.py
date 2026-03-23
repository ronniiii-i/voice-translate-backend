"""
eval_mos_generate.py — Generate TTS audio clips for the MOS survey (Table 4)

HOW TO RUN:
    From your backend/ folder:
        python evaluation/eval_mos_generate.py

WHAT IT DOES:
    Generates 5 TTS audio clips per language = 25 WAV files total.
    Saves them to: evaluation/mos_clips/
    You then share these clips with listeners via a Google Form.

AFTER RUNNING THIS SCRIPT:
    1. Upload the 25 WAV files to Google Drive (or any file sharing)
    2. Make each file shareable (Anyone with the link can view)
    3. Create a Google Form — instructions are below
    4. Send the form to 5-10 people
    5. Collect responses, compute mean and standard deviation per language
    6. Those numbers go into Table 4
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.tts_model import PiperTTS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "mos_clips")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5 sentences per language for TTS evaluation
# Keep them natural — varied length and content
TTS_SENTENCES = {
    "en": [
        "The meeting has been rescheduled to Thursday afternoon.",
        "Could you please review the attached document?",
        "We are making excellent progress on the project.",
        "The new software update will be available next week.",
        "Thank you for your patience and understanding.",
    ],
    "fr": [
        "La réunion a été reportée à jeudi après-midi.",
        "Pourriez-vous s'il vous plaît examiner le document joint?",
        "Nous faisons d'excellents progrès sur le projet.",
        "La nouvelle mise à jour logicielle sera disponible la semaine prochaine.",
        "Merci pour votre patience et votre compréhension.",
    ],
    "de": [
        "Das Meeting wurde auf Donnerstagnachmittag verschoben.",
        "Könnten Sie bitte das beigefügte Dokument prüfen?",
        "Wir machen hervorragende Fortschritte bei dem Projekt.",
        "Das neue Software-Update wird nächste Woche verfügbar sein.",
        "Vielen Dank für Ihre Geduld und Ihr Verständnis.",
    ],
    "es": [
        "La reunión ha sido reprogramada para el jueves por la tarde.",
        "¿Podría revisar el documento adjunto por favor?",
        "Estamos haciendo un excelente progreso en el proyecto.",
        "La nueva actualización de software estará disponible la próxima semana.",
        "Gracias por su paciencia y comprensión.",
    ],
    "zh": [
        "会议已改期至周四下午。",
        "请您审阅附件文件好吗？",
        "我们在这个项目上取得了很好的进展。",
        "新的软件更新将于下周推出。",
        "感谢您的耐心和理解。",
    ],
}


def main():
    print("Initialising PiperTTS...")
    tts = PiperTTS()
    print("TTS ready.\n")

    generated = []

    for lang, sentences in TTS_SENTENCES.items():
        print(f"── Generating {lang} clips ──────────────────────")
        for i, sentence in enumerate(sentences, start=1):
            filename = f"mos_{lang}_{i:02d}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)

            try:
                tts.synthesize(sentence, filepath, language=lang)
                size_kb = os.path.getsize(filepath) / 1024
                print(f"  {filename} ({size_kb:.0f} KB) — \"{sentence[:50]}\"")
                generated.append(filepath)
            except Exception as e:
                print(f"  ERROR generating {filename}: {e}")

    print(f"\n✅ Generated {len(generated)} clips in {OUTPUT_DIR}/")
    print("\nNext steps:")
    print("  1. Upload all files in evaluation/mos_clips/ to Google Drive")
    print("  2. Create a Google Form using the instructions in MOS_SURVEY_INSTRUCTIONS.txt")
    print("  3. Share the form with 5-10 people")
    print("  4. Run eval_mos_calculate.py after collecting responses")

    # Print a manifest of generated files for reference
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("MOS CLIP MANIFEST\n")
        f.write("="*50 + "\n")
        for lang, sentences in TTS_SENTENCES.items():
            for i, sentence in enumerate(sentences, start=1):
                filename = f"mos_{lang}_{i:02d}.wav"
                f.write(f"{filename}: {sentence}\n")
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()