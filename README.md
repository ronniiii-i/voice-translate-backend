# Voice Translation App

A real-time, bi-directional voice translation system designed to act as a "language bridge" between two users in a voice call.

## Features

* **Automatic Role-Swapping:** Smart routing that translates User A's native language into User B's native language and vice-versa.
* **Low Latency Core:** * **ASR:** Whisper.cpp (Tiny model) for lightning-fast speech-to-text.
* **MT:** Argos Translate (Open-source, offline Neural Machine Translation).
* **TTS:** Piper (ONNX-based) for near-instant synthesized speech.

* **Transcoding Engine:** Built-in FFmpeg integration to handle varied browser audio formats (WebM to PCM 16kHz).

---

## 🏗️ Project Structure

```text
voice-translation-app/
├── backend/
│   ├── app/
│   │   ├── models/           # Individual AI Model Wrappers
│   │   ├── services/         # Orchestration Logic (Pipeline)
│   │   └── main.py           # FastAPI WebSocket Server
│   ├── tests/                # Verification Scripts
│   └── requirements.txt      # Python Dependencies
├── frontend/
│   ├── index.html            # Web Interface
│   └── app.js                # WebSocket & Audio Logic
└── models/                   # Binary Models (Excluded from Git)
    ├── asr/                  # whisper.cpp + ggml-tiny.bin
    ├── mt/                   # argos-translate packs
    └── tts/                  # .onnx voice models

```

---

## 🛠️ Setup Instructions

### 1. System Dependencies

The backend requires **FFmpeg** for audio transcoding.

```bash
sudo apt update && sudo apt install ffmpeg

```

### 2. Python Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### 3. Model Installation (The "Brains")

This project requires specific binaries not included in the repository due to size:

* **ASR:** Compile [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and place the `main` executable and `ggml-tiny.bin` model in `models/asr/`.
* **TTS:** Download `.onnx` and `.json` voice files from the [Piper Voice Hub](https://github.com/rhasspy/piper) and place them in `models/tts/`.
* **MT:** Launch the app once; it will automatically download the required Argos language packs or use the internal management script.

---

## 🚀 Running the App

### 1. Start the Backend

```bash
cd backend
python3 app/main.py

```

Wait for the message: `✅ System Ready on Port 8000`.

### 2. Launch the Frontend (For testing purposes)

You can serve the frontend using a simple Python server:

```bash
cd frontend
python3 -m http.server 3000

```

Open `http://localhost:3000` in **two separate tabs**.

---

## 🔄 How the "Bridge" Works

1. **Handshake:** User A joins and selects "English." User B joins and selects "French."
2. **User A Speaks:** The browser sends WebM audio.
3. **Transcode:** Server converts WebM  16kHz WAV.
4. **Process:** Audio  Text(EN)  Text(FR)  Speech(FR).
5. **Routing:** The server identifies User B as the peer and sends the French audio to their WebSocket only.
