# Voice Translation App - AI Coding Agent Instructions

## Architecture (big picture)

- Pipeline: ASR → MT → TTS coordinated by `TranslationPipeline` in [backend/app/services/translation_pipeline.py](backend/app/services/translation_pipeline.py).
- ASR uses `whisper.cpp` CLI via `WhisperASR` in [backend/app/models/asr_model.py](backend/app/models/asr_model.py) and reads {audio}.json output.
- MT uses Argos via `ArgosTranslator` in [backend/app/models/mt_model.py](backend/app/models/mt_model.py).
- TTS uses Piper ONNX via `PiperTTS` in [backend/app/models/tts_model.py](backend/app/models/tts_model.py).

## API surface

- REST: POST /translate in [backend/app/main.py](backend/app/main.py) returns WAV plus X-Source-Text and X-Translated-Text headers.
- WebSocket: /ws/translate in [backend/app/api/websocket_routes.py](backend/app/api/websocket_routes.py) expects JSON config → binary audio chunks → process → get_audio.

## Dev workflows (non-obvious)

- Build whisper.cpp binary before ASR: see [models/asr/whisper.cpp/README.md](models/asr/whisper.cpp/README.md).
- Local server: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 from backend.
- Docker: docker-compose up --build mounts ./models to /app/models.
- Tests live in [backend/tests](backend/tests): [backend/tests/test_asr.py](backend/tests/test_asr.py) uses ffprobe; [backend/tests/test_pipeline.py](backend/tests/test_pipeline.py) targets <3x realtime for 5s audio.

## Project-specific patterns

- ASR errors return strings (don’t raise) in `WhisperASR.transcribe()` to keep pipeline running.
- `translate_long_text()` chunks at 512 chars using sentence boundaries in [backend/app/services/translation_pipeline.py](backend/app/services/translation_pipeline.py).
- Temporary files use tempfile.mktemp() and cleanup in finally blocks in [backend/app/main.py](backend/app/main.py).
- WebSocket buffers audio in bytearray() until action=process; action=get_audio streams the TTS WAV.

## Integration points

- whisper.cpp CLI flags -nt and -oj are required for JSON sidecar output.
- Piper models live in [models/tts](models/tts) with matching .onnx.json; language mapping in `PiperTTS.model_map` in [backend/app/models/tts_model.py](backend/app/models/tts_model.py).
- Language codes are ISO 639-1 (e.g., en, fr, es).

## Frontend

- [frontend](frontend) is empty; API-only backend with docs at /docs.

# Backend dependencies
