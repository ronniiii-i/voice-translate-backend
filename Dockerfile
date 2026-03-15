# ──────────────────────────────────────────────────────────────────────────────
# LinguaCall Backend — Production Dockerfile
# Target: Hugging Face Spaces (free tier) or any Docker host
#
# What this builds:
#   - Python 3.11 slim base
#   - System deps: ffmpeg, build tools, piper-tts binary
#   - All Python deps from requirements.txt
#   - Argos Translate + all direct language pairs (en/fr/de/es/zh ↔ each other)
#   - Piper voice models (low quality) for en, fr, de, es, zh
#   - Whisper model downloaded at runtime (not build time) to keep image size down
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
# Piper is a standalone binary — we pull the latest release directly
RUN wget -q https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
  -O /tmp/piper.tar.gz \
  && tar -xzf /tmp/piper.tar.gz -C /usr/local/bin/ --strip-components=1 \
  && rm /tmp/piper.tar.gz \
  && piper --help > /dev/null 2>&1 || true

# ── 3. Set working directory ──────────────────────────────────────────────────
WORKDIR /app

# ── 4. Python dependencies ────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer independently
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# ── 5. Download Argos Translate language pairs ────────────────────────────────
# Languages: English (en), French (fr), German (de), Spanish (es), Chinese (zh)
#
# Direct pairs that exist in Argos package index:
#   en↔fr, en↔de, en↔es, en↔zh
#   fr↔de, fr↔es
#   de↔es
#
# Pairs like zh↔fr, zh↔de, zh↔es don't exist as direct packages —
# Argos will automatically pivot through English at runtime. No action needed.
RUN python3 << 'EOF'
import argostranslate.package

PAIRS = [
("en", "fr"), ("fr", "en"),
("en", "de"), ("de", "en"),
("en", "es"), ("es", "en"),
("en", "zh"), ("zh", "en"),
("fr", "de"), ("de", "fr"),
("fr", "es"), ("es", "fr"),
("de", "es"), ("es", "de"),
]

print("Updating Argos package index...")
argostranslate.package.update_package_index()
available = argostranslate.package.get_available_packages()
available_map = {(p.from_code, p.to_code): p for p in available}

for src, tgt in PAIRS:
pkg = available_map.get((src, tgt))
if pkg:
print(f"Installing {src} -> {tgt}...")
argostranslate.package.install_from_path(pkg.download())
else:
print(f"Skipping {src} -> {tgt} (no direct package, will pivot via English)")

print("Done.")
'EOF'

# ── 6. Download Piper voice models (low quality) ──────────────────────────────
# Low quality = smaller files, faster synthesis, fits free tier RAM budget
RUN mkdir -p /app/models/tts

# English — ryan low
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx" \
  -O /app/models/tts/en_US-ryan-low.onnx \
  && wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx.json" \
  -O /app/models/tts/en_US-ryan-low.onnx.json

# French — siwis low
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/low/fr_FR-siwis-low.onnx" \
  -O /app/models/tts/fr_FR-siwis-low.onnx \
  && wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/low/fr_FR-siwis-low.onnx.json" \
  -O /app/models/tts/fr_FR-siwis-low.onnx.json

# German — thorsten low
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx" \
  -O /app/models/tts/de_DE-thorsten-low.onnx \
  && wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx.json" \
  -O /app/models/tts/de_DE-thorsten-low.onnx.json

# Spanish — mls low
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx" \
  -O /app/models/tts/es_ES-mls_10246-low.onnx \
  && wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx.json" \
  -O /app/models/tts/es_ES-mls_10246-low.onnx.json

# Chinese — huayan low
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/low/zh_CN-huayan-low.onnx" \
  -O /app/models/tts/zh_CN-huayan-low.onnx \
  && wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/low/zh_CN-huayan-low.onnx.json" \
  -O /app/models/tts/zh_CN-huayan-low.onnx.json

# ── 7. Copy application code ──────────────────────────────────────────────────
COPY backend/app ./app

# ── 8. Environment variables ──────────────────────────────────────────────────
ENV MODEL_DIR=/app/models
ENV PYTHONPATH=/app
# Tell faster-whisper to cache downloaded models here
ENV HF_HOME=/app/models/whisper_cache
# Hugging Face Spaces runs as port 7860 by default
ENV PORT=7860

# ── 9. Expose port ────────────────────────────────────────────────────────────
# HF Spaces expects 7860. For other hosts change to 8000.
EXPOSE 7860

# ── 10. Start server ──────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]