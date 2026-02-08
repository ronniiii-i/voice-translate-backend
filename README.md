# 🌐 Voice Translation Bridge

A real-time, bi-directional voice translation system designed to act as a seamless "language bridge" between two users. It detects speech automatically, translates it, and plays the synthesized voice to the peer in their native language.

## ✨ Features

* **Hands-Free VAD:** Automatic Voice Activity Detection (VAD) detects when you stop speaking—no "Push-to-Talk" needed.
* **Intelligent Routing:** Automatically identifies the peer in the room and routes the translated audio to them.
* **Full Offline-Capable Stack:**
* **ASR:** [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) (Tiny model) for high-speed speech-to-text.
* **MT:** [Argos Translate](https://github.com/argosopentech/argos-translate) for open-source, offline Neural Machine Translation.
* **TTS:** [Piper](https://github.com/rhasspy/piper) (ONNX-based) for near-instant, human-like synthesized speech.
* **PCM Streaming:** Low-latency raw audio streaming via Web Audio API.

---

## 🏗️ Project Structure

```text
voice-translation-app/
├── backend/
│   ├── app/
│   │   ├── services/         # VAD processing & Translation Pipeline
│   │   └── main.py           # FastAPI WebSocket Server
│   └── requirements.txt      # Python Dependencies
├── frontend/
│   └── index.html            # Web Interface (PCM Audio Logic)
├── models/                   # AI Models (Stored locally)
│   ├── asr/                  # whisper.cpp source + tiny model
│   ├── mt/                   # argos-translate language packs
│   └── tts/                  # .onnx + .json voice models
└── scripts/                  
    └── setup_models.sh       # Automation script to build/download models

```

---

## 🛠️ Setup Instructions

### 1. System Dependencies

The backend requires **FFmpeg** for final audio containerization and **CMake/Build-Essentials** to compile the ASR engine.

```bash
sudo apt update && sudo apt install ffmpeg build-essential cmake

```

### 2. Automated Model Setup

We provide a setup script that clones `whisper.cpp`, compiles it, downloads the Piper voices, and installs Argos language packs.

```bash
chmod +x setup_models.sh
./setup_models.sh

```

### 3. Python Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

---

## 🚀 Running the App

### 1. Start the Backend

```bash
# From the backend directory
python3 app/main.py

```

Wait for: `✅ Models Warm. System Ready on Port 8000.`

### 2. Launch the Frontend

Serve the frontend using a simple server:

```bash
cd frontend
python3 -m http.server 3000

```

1. Open `http://localhost:3000` in **Tab 1** (Set to English).
2. Open `http://localhost:3000` in **Tab 2** (Set to French).
3. Speak naturally in Tab 1; the translated audio will play automatically in Tab 2.

---

## 🔄 How the "Bridge" Works

1. **PCM Streaming:** The browser captures audio at 16kHz and sends raw PCM bytes via WebSocket.
2. **VAD Analysis:** The server analyzes incoming chunks. When ~1.2s of silence is detected, it triggers the pipeline.
3. **The Pipeline:**
    * **ASR:** `whisper.cpp` converts the buffered PCM into English text.
    * **MT:** `Argos` translates English text to French.
    * **TTS:** `Piper` generates a French `.wav` file from the translated text.

4. **Targeted Delivery:** The server identifies the peer in the room and sends the `.wav` bytes + JSON captions to **only** that user.
