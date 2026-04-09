<<<<<<< HEAD
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (FFmpeg for Whisper/Deepgram, eSpeak for text-to-speech)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run the FastAPI server directly on 0.0.0.0
EXPOSE 9067
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9067"]
=======
# syntax=docker/dockerfile:1
# Use Python 3.11 (Compatible with faster-whisper)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv for faster package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install system dependencies
# ffmpeg: for video processing
# espeak-ng: for Kokoro TTS phonemizer
# git: for installing git dependencies if any
# build-essential & portaudio19-dev: for PyAudio/SoundDevice
# curl: for healthchecks
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PHONEMIZER_ESPEAK_PATH=/usr/bin/espeak-ng
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy the rest of the application
COPY . .

# Create temp directory
RUN mkdir -p temp

# Expose the port
EXPOSE 8000

# Run the application
# We use python main.py to trigger the startup logic (model checks)
CMD ["python", "main.py"]
>>>>>>> 4a061bd5aa45e150fe8069b75d008d668a073d2e
