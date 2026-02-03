
# Fluency-Net

A real-time, AI-powered speech therapy assistant designed to help users improve fluency. This application uses advanced speech-to-speech technology to analyze stuttering patterns, provide therapeutic feedback, and generate fluent audio/video output.

## 🌟 Features

- **Real-Time Streaming**: Low-latency speech analysis using WebSockets and AudioWorklet.
- **Multilingual Support**: Supports English, Hindi, Telugu, Spanish, French, and many more.
- **AI Analysis**: Uses **Ollama (Llama 3.1)** and **Agno** to detect stuttering types (Repetitions, Blocks, Prolongations) and provide clinical SOAP notes.
- **Adaptive Agentic Workflow**: Implements a stateful reflex agent that dynamically adjusts therapy goals (e.g., switching from "Fluency Shaping" to "Anxiety Reduction") based on real-time user performance metrics.
- **Fluent Regeneration**: Reconstructs fragmented speech into fluent text and audio using **Kokoro TTS** (High Quality) or **Edge TTS**.
- **Acoustic Features**: Analyzes RMS Energy and Zero Crossing Rate to detect physical tension and struggle behaviors.

## 🛠️ Prerequisites

Before running the application, ensure you have the following installed:

1. **Python 3.10, 3.11, or 3.12** (Python 3.13+ is currently incompatible with `faster-whisper`).
2. **FFmpeg**: Required for video processing.
    - Windows: `winget install Gyan.FFmpeg`
    - Mac: `brew install ffmpeg`
    - Linux: `sudo apt install ffmpeg`
3. **Ollama**: Required for the AI Agent logic.
    - Download from ollama.com.
    - Pull the model: `ollama pull llama3.1:8b`

## 🚀 Installation & Setup

1. **Clone the Repository**

    ```bash
    git clone <your-repo-url>
    cd Stutter2Fluent
    ```

2. **Create a Virtual Environment** (Recommended)

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3. **Run the Setup Script**
    This script installs Python dependencies and downloads necessary AI models (Kokoro ONNX).

    ```bash
    python download_requirements.py
    ```

4. **Start Ollama**
    Ensure Ollama is running in the background.

    ```bash
    ollama serve
    ```

## ▶️ Running the Application

Start the FastAPI server:

```bash
python main.py
```

- The application will launch at `http://localhost:8000`.
- If port 8000 is busy, you can change it by setting the `PORT` environment variable.

## 🐳 Docker Deployment

You can run the entire stack (App + Ollama) using Docker Compose.

1. **Build and Start Services**

    ```bash
    docker-compose up -d --build
    ```

2. **Download the AI Model** (First time only)
    Since the Ollama container starts empty, you need to pull the model:

    ```bash
    docker exec -it fluency-net-ollama ollama pull llama3.1:8b
    ```

3. **Access the App**
    Open `http://localhost:8000` in your browser.

## 📂 Project Structure

- `main.py`: Core backend logic (FastAPI, WebSocket, Audio Pipeline).
- `index.html`: Frontend UI (Bento Grid design, Audio Recording, Streaming).
- `download_requirements.py`: Helper script for setup.
- `requirements.txt`: Python dependency list.
- `temp/`: Stores temporary audio/video files during processing (auto-cleaned).

## ⚙️ Configuration

You can configure the application using environment variables or a `.env` file:

- `OLLAMA_HOST`: URL of the Ollama server (default: `http://127.0.0.1:11434`).
- `PORT`: Port to run the web server on (default: `8000`).

## 🧠 How It Works

### System Architecture

1. **Multimodal Ingestion**: Fuses **Faster-Whisper** transcripts with raw acoustic data (RMS Energy, ZCR) to detect non-verbal "blocks" that text-only models miss.
2. **Stateful Agent Brain**: Uses `AgentInternalState` to track conversation history and emotional trajectory. The system employs a **Goal-Based Strategy** engine that evaluates the success of previous interventions before selecting the next therapeutic tactic.
3. **Local-First Inference**: Runs entirely on-device using **Ollama** and **Int8 Quantization**, ensuring patient data privacy and zero cloud latency.
4. **Generative Synthesis**: Reconstructs fluent speech using **Kokoro TTS** (ONNX) for natural prosody restoration.

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/NewFeature`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

Here is a comprehensive breakdown of the features, techniques, functions, modules, libraries, methods, and algorithms used in the Fluency-Net (Stutter2Fluent) application code provided.

1. Core Features
Real-Time Speech-to-Speech Pipeline: Converts input audio to text, processes it via AI, and synthesizes fluent speech output.
Multilingual Support: Explicit support for 13 languages (English, Hindi, Telugu, Bengali, Kannada, Tamil, Malayalam, Spanish, French, German, Chinese, Japanese, Russian) with specific handling for Indian languages.
Clinical Stuttering Analysis: Detects specific dysfluencies (Repetitions, Blocks, Prolongations, Interjections) and generates professional SOAP Notes (Subjective, Objective, Assessment, Plan).
Adaptive AI Agent:
Model-Based Reflex Agent: Maintains an internal state of the user (history, emotional state, dysfluency profile).
Goal-Based Strategy: Dynamically switches therapy goals (e.g., "Anxiety Reduction" vs. "Fluency Shaping") based on user performance.
Acoustic Analysis: Extracts real-time audio features like Pitch, Loudness (RMS), and Noisiness (ZCR) to aid clinical diagnosis.
Hybrid TTS System:
Kokoro (ONNX): High-quality local TTS for English.
Edge-TTS: Cloud-based TTS for multilingual support.
Session Persistence: Saves user progress and agent state to a local SQLite database.
Robust Error Handling: Includes fallbacks for audio decoding (FFmpeg), JSON parsing, and TTS generation.
2. Libraries & Modules
Standard Library:

os, sys, shutil: File system and environment management.
asyncio: Asynchronous programming for non-blocking I/O.
json, ast: Data parsing (JSON and Python literals).
sqlite3: Lightweight database for session storage.
logging: System-wide logging.
subprocess: Running external commands (FFmpeg, Ollama).
re: Regular expressions for text cleaning and parsing.
hashlib, uuid: Generating unique IDs and cache keys.
io, base64: Binary data handling.
Third-Party Libraries:

Web Framework:
fastapi: High-performance API framework.
uvicorn: ASGI server implementation.
jinja2: HTML templating engine.
AI & Speech:
faster_whisper: Optimized implementation of OpenAI's Whisper model for STT.
kokoro_onnx: ONNX runtime for the Kokoro TTS model.
edge_tts: Python wrapper for Microsoft Edge's online TTS service.
agno (formerly Phidata): Framework for building AI agents.
ollama: Client for interacting with local LLMs (Llama 3.1).
langdetect: Heuristic-based language detection.
jiwer: Library for calculating Word Error Rate (WER).
Data Science & Audio:
numpy: Numerical computing for audio signal processing.
scipy: WAV file reading/writing.
pydantic: Data validation and settings management.
requests: HTTP library for downloading models/files.
3. Algorithms & Techniques
Speech Processing:

Beam Search Decoding: Used in Whisper (beam_size=5) to explore multiple transcription paths, crucial for accuracy in agglutinative languages like Telugu.
Voice Activity Detection (VAD): Filters out silence using energy thresholds (min_silence_duration_ms=2000) to focus processing on active speech.
Autocorrelation: Used in extract_acoustic_features to estimate the Fundamental Frequency (Pitch/F0) of the voice.
Zero Crossing Rate (ZCR): Algorithm to calculate the noisiness of the signal, helping detect fricative sounds (s, sh, f).
Root Mean Square (RMS): Calculates the average power (loudness) of the audio signal.
AI & LLM Techniques:

Chain-of-Thought (CoT) Prompting: The system prompt instructs the LLM to output <thought> tags before the JSON, forcing it to reason step-by-step before generating the final analysis.
Few-Shot Learning: The prompt includes specific examples (Input -> Output pairs) for multiple languages to "teach" the model the expected behavior.
Heuristic Language Detection: A custom algorithm that combines Whisper's detection, langdetect library, and script analysis (Latin vs. Non-Latin ratio) to accurately identify the language.
Software Engineering:

Dependency Injection: Used via FastAPI's Depends (implicit in route handlers).
Caching Strategy: TTS audio is cached using an MD5 hash of text + voice + speed to prevent redundant processing.
Retry Logic: Implemented in knowledge_agent_client to handle transient LLM failures.
4. Key Functions & Methods
process_audio_pipeline: The central orchestrator that manages the flow from Audio Upload -> Transcription -> AI Analysis -> TTS -> Response.
knowledge_agent_client: Manages the interaction with the LLM, including prompt construction and JSON extraction.
extract_json_from_text: A robust parser that cleans raw LLM output (removing markdown, fixing trailing commas) to extract valid JSON.
determine_agent_goal: Implements the logic for the Goal-Based Agent, deciding whether to focus on "Anxiety Reduction" or "Fluency Shaping".
tts_router: Intelligently routes TTS requests to either Kokoro (for high-quality English) or Edge-TTS (for other languages), handling voice selection and fallbacks.
extract_acoustic_features: Computes RMS, ZCR, Variance, and Pitch from raw numpy audio arrays.
decode_audio_fallback: Uses system FFmpeg via subprocess to decode audio if the primary library fails.
check_and_download_models: Ensures required model files (ONNX) are present on startup.
5. Specifications
Audio Sample Rate: 16,000 Hz (Standard for speech recognition models).
Whisper Model: Configurable tiers (base, small, large-v3).
LLM Model: Llama 3.1 8B (via Ollama).
Database: SQLite (sessions.db) storing JSON-serialized state.
Containerization: Dockerfile based on python:3.11-slim with uv for fast package management.

