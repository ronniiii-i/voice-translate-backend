#!/bin/bash

# Exit on error
set -e

echo "🏗️  Creating directory structure..."
PROJECT_ROOT=$(pwd)
mkdir -p ./models/asr ./models/tts ./models/mt

# --- 1. WHISPER.CPP SETUP (ASR) ---
echo "📥 Setting up Whisper.cpp (ASR)..."
cd "$PROJECT_ROOT/models/asr"

if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp.git
fi

cd whisper.cpp
# Download the tiny model using their helper script
bash ./models/download-ggml-model.sh tiny

echo "🛠️  Building whisper.cpp..."
cmake -B build
cmake --build build -j --config Release

# Verify build
if [ -f "./build/bin/whisper-cli" ]; then
    echo "✅ Whisper.cpp built successfully."
else
    echo "❌ Whisper.cpp build failed."
    exit 1
fi

# --- 2. PIPER VOICES (TTS) ---
echo "📥 Downloading Piper Voice models (TTS)..."
cd "$PROJECT_ROOT/models/tts"

# French Voice (Siwis Medium)
curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx -o fr_FR-siwis-medium.onnx
curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json -o fr_FR-siwis-medium.onnx.json

# English Voice (Bryce Medium)
curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/bryce/medium/en_US-bryce-medium.onnx -o en_US-bryce-medium.onnx
curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/bryce/medium/en_US-bryce-medium.onnx.json -o en_US-bryce-medium.onnx.json

# --- 3. ARGOS TRANSLATE (MT) ---
echo "🤖 Installing Argos Language Packs (MT)..."
# Ensure we are in the backend venv if it exists
if [ -d "$PROJECT_ROOT/backend/venv" ]; then
    source "$PROJECT_ROOT/backend/venv/bin/activate"
fi

pip install argostranslate

python3 << EOF
import argostranslate.package
print("Updating Argos package index...")
argostranslate.package.update_package_index()
available = argostranslate.package.get_available_packages()

pairs = [("en", "fr"), ("fr", "en")]
for f, t in pairs:
    print(f"Installing {f} -> {t}...")
    pkg = next(filter(lambda x: x.from_code == f and x.to_code == t, available))
    argostranslate.package.install_from_path(pkg.download())
EOF

# --- 4. SYSTEM DEPENDENCIES CHECK ---
echo "🔍 Checking system dependencies..."
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  WARNING: FFmpeg not found. Please install it: sudo apt install ffmpeg"
else
    echo "✅ FFmpeg is installed."
fi

echo "-----------------------------------------------"
echo "✅ Setup Complete! All models prepared."
echo "🚀 You can now start the backend and frontend."