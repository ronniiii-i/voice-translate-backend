# ──────────────────────────────────────────────────────────────────────────────
# LinguaCall Backend — Production Dockerfile
# Target: Hugging Face Spaces (free tier) or any Docker host
#
# What this builds:
#   - Python 3.11 slim base
#   - System deps: ffmpeg, build tools, piper-tts binary
#   - All Python deps from requirements.txt (includes transformers + torch)
#   - Piper voice models (low/x-low quality) for en, fr, de, es, zh
#   - Helsinki-NLP OPUS-MT models download at runtime on first use (lazy load)
#   - faster-whisper base model downloads at runtime on first startup
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── 1. System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
  ffmpeg \
  build-essential \
  cmake \
  git \
  curl \
  wget \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# ── 2. Install Piper TTS binary ───────────────────────────────────────────────
RUN wget -q https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
  -O /tmp/piper.tar.gz \
  && tar -xzf /tmp/piper.tar.gz -C /usr/local/bin/ --strip-components=1 \
  && rm /tmp/piper.tar.gz \
  && piper --help > /dev/null 2>&1 || true

# ── 3. Set working directory ──────────────────────────────────────────────────
WORKDIR /app

# ── 4. Python dependencies ────────────────────────────────────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# ── 5. Download Piper voice models (low / x-low quality) ─────────────────────
RUN mkdir -p /app/models/tts

# English — ryan low
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx" \
  -O /app/models/tts/en_US-ryan-low.onnx \
  && wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx.json" \
  -O /app/models/tts/en_US-ryan-low.onnx.json

# French — siwis low
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/low/fr_FR-siwis-low.onnx" \
  -O /app/models/tts/fr_FR-siwis-low.onnx \
  && wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/low/fr_FR-siwis-low.onnx.json" \
  -O /app/models/tts/fr_FR-siwis-low.onnx.json

# German — thorsten low
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx" \
  -O /app/models/tts/de_DE-thorsten-low.onnx \
  && wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx.json" \
  -O /app/models/tts/de_DE-thorsten-low.onnx.json

# Spanish — mls_10246 low
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx" \
  -O /app/models/tts/es_ES-mls_10246-low.onnx \
  && wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx.json" \
  -O /app/models/tts/es_ES-mls_10246-low.onnx.json

# Chinese — huayan x-low (only available quality in Piper voice library)
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx" \
  -O /app/models/tts/zh_CN-huayan-x_low.onnx \
  && wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx.json" \
  -O /app/models/tts/zh_CN-huayan-x_low.onnx.json

# ── 6. Copy application code ──────────────────────────────────────────────────
COPY backend/app ./app

# ── 7. Environment variables ──────────────────────────────────────────────────
ENV MODEL_DIR=/app/models
ENV PYTHONPATH=/app
ENV HF_HOME=/app/models/whisper_cache
ENV PORT=7860

# ── 8. Expose port ────────────────────────────────────────────────────────────
EXPOSE 7860

# ── 9. Start server ───────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]