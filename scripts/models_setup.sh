#!/bin/bash

# Exit on error
set -e

echo "🏗️  Creating directory structure..."
PROJECT_ROOT=$(pwd)
mkdir -p ./models/tts

# --- 1. PIPER TTS BINARY ---
echo "📥 Installing Piper TTS binary..."
if ! command -v piper &> /dev/null; then
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
    wget -q "$PIPER_URL" -O /tmp/piper.tar.gz
    sudo tar -xzf /tmp/piper.tar.gz -C /usr/local/bin/ --strip-components=1
    rm /tmp/piper.tar.gz
    echo "✅ Piper installed."
else
    echo "✅ Piper already installed."
fi

# --- 2. PIPER VOICE MODELS (TTS) ---
echo "📥 Downloading Piper voice models (low quality)..."
cd "$PROJECT_ROOT/models/tts"

# English — ryan low
wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx" \
    -O en_US-ryan-low.onnx
wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx.json" \
    -O en_US-ryan-low.onnx.json
echo "  ✅ English (ryan low)"

# French — siwis low
wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/low/fr_FR-siwis-low.onnx" \
    -O fr_FR-siwis-low.onnx
wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/low/fr_FR-siwis-low.onnx.json" \
    -O fr_FR-siwis-low.onnx.json
echo "  ✅ French (siwis low)"

# German — thorsten low
wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx" \
    -O de_DE-thorsten-low.onnx
wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx.json" \
    -O de_DE-thorsten-low.onnx.json
echo "  ✅ German (thorsten low)"

# Spanish — mls_10246 low
wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx" \
    -O es_ES-mls_10246-low.onnx
wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx.json" \
    -O es_ES-mls_10246-low.onnx.json
echo "  ✅ Spanish (mls_10246 low)"

# Chinese — huayan x_low (only available quality)
wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx" \
    -O zh_CN-huayan-x_low.onnx
wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx.json" \
    -O zh_CN-huayan-x_low.onnx.json
echo "  ✅ Chinese (huayan x_low)"

# --- 3. PYTHON ENVIRONMENT ---
echo "🐍 Setting up Python environment..."
cd "$PROJECT_ROOT/backend"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created."
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Python dependencies installed."

# --- 4. SYSTEM DEPENDENCIES CHECK ---
echo "🔍 Checking system dependencies..."

if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  WARNING: FFmpeg not found. Install it with: sudo apt install ffmpeg"
else
    echo "✅ FFmpeg found."
fi

if ! command -v wget &> /dev/null; then
    echo "⚠️  WARNING: wget not found. Install it with: sudo apt install wget"
else
    echo "✅ wget found."
fi

echo "-----------------------------------------------"
echo "✅ Setup complete! All models and language packs ready."
echo ""
echo "🚀 To start the backend:"
echo "   cd backend && source venv/bin/activate"
echo "   python3 -m app.main"
echo ""
echo "⚠️  NOTE: Whisper model (faster-whisper 'base') downloads automatically"
echo "   on first startup (~150MB). Subsequent starts use the cached version."