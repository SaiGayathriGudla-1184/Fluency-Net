
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

## 📄 License

MIT License

```

<!--
[PROMPT_SUGGESTION]How can I create a Dockerfile to containerize this application for easier deployment?[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]Explain how to set up a GitHub Action to automatically lint the Python code on push.[/PROMPT_SUGGESTION]
