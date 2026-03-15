---
title: LinguaCall Backend
emoji: 🌐
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🌐 LinguaCall — Real-Time Voice Translation Backend

LinguaCall is a real-time, bi-directional voice translation system that acts as a seamless language bridge between two users. It automatically detects speech, transcribes it, translates it, and plays synthesized audio to the other person in their native language — all with no push-to-talk required.

---

## ✨ Features

- **Hands-Free VAD** — Voice Activity Detection automatically triggers the pipeline when you stop speaking (~1.2s silence). No button needed.
- **Bi-Directional Translation** — Both users speak in their own language simultaneously. Each hears the other translated in real time.
- **5 Supported Languages** — English, French, German, Spanish, and Chinese in any combination.
- **Synchronized Captions** — Translated text appears in sync with audio playback, not when processing finishes.
- **Hallucination Filtering** — ASR output is filtered for Whisper hallucinations (loops, silence artifacts, subtitle injections) before translation.
- **Drop-on-Busy** — If a pipeline is already running for a user, new segments are dropped rather than queued, preventing cascading lag.
- **Fully Offline-Capable Stack** — All AI components run locally with no external API calls.

---

## 🏗️ Architecture

```
Browser (React Frontend)
        │
        │  WebSocket (PCM16 audio @ 16kHz)
        ▼
FastAPI Backend  ──────────────────────────────────────────────
        │
        ├── StreamingVAD         # RMS energy-based speech detection
        │
        └── TranslationPipeline
              ├── WhisperASR     # faster-whisper (base, CPU int8)
              ├── ArgosTranslator # Argos Translate (offline NMT)
              └── PiperTTS       # Piper ONNX (persistent subprocess)
```

### WebSocket Protocol

```
Client → Server:  JSON handshake  { "native_lang": "en" }
Client → Server:  Binary PCM16 chunks (4096 samples @ 16kHz, ~256ms each)
Client → Server:  JSON ping       { "type": "ping" }

Server → Client:  JSON            { "type": "connected", "user_id": "...", "room": "..." }
Server → Client:  JSON            { "type": "peer_joined", "peer_id": "..." }
Server → Client:  JSON            { "type": "peer_left",   "peer_id": "..." }
Server → Client:  Binary framed   [4-byte JSON length][JSON metadata][WAV bytes]
                                   metadata: { "type": "audio_with_caption", "text": "...", "original": "..." }
```

---

## 🤖 AI Stack

| Component | Library | Model | Notes |
|-----------|---------|-------|-------|
| ASR | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `base` (CPU int8) | ~2–4s on modern CPU |
| MT | [Argos Translate](https://github.com/argosopentech/argos-translate) | Offline NMT packages | Direct pairs + English pivot |
| TTS | [Piper](https://github.com/rhasspy/piper) | Low quality ONNX voices | Persistent subprocess per language |

### Supported Language Pairs

Direct Argos packages are installed for all available combinations. Pairs without a direct package (e.g. `zh↔de`) automatically pivot through English at runtime.

| | EN | FR | DE | ES | ZH |
|---|---|---|---|---|---|
| **EN** | — | ✅ | ✅ | ✅ | ✅ |
| **FR** | ✅ | — | ✅ | ✅ | → EN |
| **DE** | ✅ | ✅ | — | ✅ | → EN |
| **ES** | ✅ | ✅ | ✅ | — | → EN |
| **ZH** | ✅ | → EN | → EN | → EN | — |

---

## 🚀 API Reference

### `GET /health`
Returns server status and active rooms.

```json
{
  "status": "ok",
  "rooms": {
    "room1": ["user_abc123", "user_def456"]
  }
}
```

### `GET /rooms/{room_id}`
Check if a room exists and how many users are in it.

```json
{
  "exists": true,
  "room_id": "room1",
  "occupants": 1
}
```

### `WS /ws/call/{room_id}/{user_id}`
Main WebSocket endpoint. See protocol above.

- Rooms support a maximum of **2 users**
- Attempting to join a full room returns a `4003` close code
- Each user must send a handshake JSON within **10 seconds** of connecting

---

## 🛠️ Local Development

### Prerequisites

```bash
sudo apt install ffmpeg build-essential cmake
```

### 1. Download Models

```bash
chmod +x scripts/models_setup.sh
./scripts/models_setup.sh
```

This downloads and compiles whisper.cpp, downloads Piper voice models, and installs Argos language packs.

### 2. Python Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Backend

```bash
# from project root
python3 -m app.main
```

Wait for: `✅ System Ready!`

### 4. Run the Test Frontend

```bash
cd frontend
python3 -m http.server 3000
```

Open `http://localhost:3000` in two tabs, set different languages, same Room ID, and speak.

---

## 🐳 Docker

### Build & Run

```bash
docker build -t linguacall-backend .
docker run -p 7860:7860 linguacall-backend
```

### Docker Compose (local dev with frontend)

```bash
docker-compose up --build
```

---

## ☁️ Deployment (Hugging Face Spaces)

This repo is configured for direct deployment to Hugging Face Spaces using the Docker SDK.

1. Create a new Space → SDK: **Docker**
2. Add this repo as a remote and push:
```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/linguacall-backend
git push space main
```
3. Monitor the build in the **Logs** tab (~10–15 min first build)
4. Once live, your backend is at:
```
https://YOUR_USERNAME-linguacall-backend.hf.space
```
5. Connect your frontend using:
```
wss://YOUR_USERNAME-linguacall-backend.hf.space
```

---

## 📁 Project Structure

```
linguacall-backend/
├── Dockerfile                        # Production build (HF Spaces / any Docker host)
├── docker-compose.yml                # Local dev with nginx frontend
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py                   # FastAPI app, WebSocket handler, room management
│       ├── logger.py                 # Rotating file + console logger
│       ├── models/
│       │   ├── asr_model.py          # WhisperASR + hallucination filtering
│       │   ├── mt_model.py           # ArgosTranslator
│       │   └── tts_model.py          # PiperTTS with persistent subprocesses
│       └── services/
│           ├── translation_pipeline.py  # ASR → MT → TTS orchestration
│           └── vad_processor.py         # RMS energy VAD with adaptive thresholding
├── frontend/
│   └── index.html                    # Standalone test client (not for production)
├── scripts/
│   └── models_setup.sh               # One-shot model download + compile script
└── models/                           # AI models (git-ignored, built at Docker build time)
    ├── asr/
    ├── mt/
    └── tts/
```

---

## ⚙️ Configuration

Key parameters in `main.py` and `vad_processor.py` you may want to tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `silence_threshold` | `1.2s` | Silence duration before triggering pipeline |
| `min_speech_duration` | `0.8s` | Minimum speech length to process (filters short blips) |
| `energy_threshold` | `800.0` | RMS energy threshold for speech detection |
| `max_speech_duration` | `15.0s` | Hard cap before force-processing |
| `MAX_USERS_PER_ROOM` | `2` | Max concurrent users per room |
| Whisper model size | `base` | Change to `tiny` if RAM-constrained |

---

## 🔧 Troubleshooting

**OOM crash on startup (Hugging Face Spaces)**
Switch Whisper to `tiny` in `backend/app/services/translation_pipeline.py`:
```python
self.asr = WhisperASR(model_size="tiny")
```

**VAD not triggering / triggering too often**
Watch the `[VAD] avg=... peak=...` logs and adjust `energy_threshold` in `main.py`. Typical values: quiet mic ~250–400, normal room ~600–1000, noisy environment ~1200+.

**Piper TTS producing no audio**
Check that `.onnx` and `.onnx.json` files are both present in `models/tts/`. Both files are required per voice.

**WebSocket disconnecting on Hugging Face**
HF Spaces has a proxy timeout. The frontend sends a ping every 25s to keep the connection alive — make sure your frontend implements this keepalive.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.