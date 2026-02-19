# Use a Python image with build tools
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
  ffmpeg build-essential cmake git curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all files
COPY . .

# Run your setup script to build whisper.cpp and download models
RUN chmod +x setup_models.sh && ./setup_models.sh

# Install Python requirements
RUN pip install --no-cache-dir -r backend/requirements.txt

# Set Python path
ENV PYTHONPATH=/app/backend

# Start the app
CMD ["python3", "-m", "app.main"]