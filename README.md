# Vocal Agent - Real-Time Speech-to-Speech AI Assistant 🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-green.svg)](https://ollama.com/)
A sophisticated real-time voice assistant that seamlessly integrates speech recognition, AI reasoning, and neural text-to-speech synthesis. Designed for natural conversational interactions with advanced tool-calling capabilities.

## 🔄 How Vocal Agent Works

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VOCAL AGENT WORKFLOW                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

    🎤 USER SPEAKS
         │
         ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   AUDIO CAPTURE     │    │   VOICE ACTIVITY     │    │  SPEECH-TO-TEXT     │
│                     │───▶│     DETECTION        │───▶│                     │
│ • Microphone Input  │    │ • Silero VAD         │    │ • Whisper large-v1  │
│ • 16kHz Sampling    │    │ • Real-time Monitor  │    │ • Language: English │
│ • Continuous Stream │    │ • Start/Stop Detect  │    │ • CUDA Acceleration │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                                                 │
                                                                 ▼
                                                    📝 "What's the weather in Tokyo?"
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AI REASONING ENGINE                                │
│                                                                                 │
│  ┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐   │
│  │   LLAMA 3.1 8B  │    │    AGNO FRAMEWORK    │    │   TOOL SELECTION    │   │
│  │                 │───▶│                      │───▶│                     │   │
│  │ • Via Ollama    │    │ • Agent Orchestration│    │ • Google Search     │   │
│  │ • Local LLM     │    │ • Context Management │    │ • Wikipedia         │   │
│  │ • 8B Parameters │    │ • Response Generation│    │ • ArXiv Papers      │   │
│  └─────────────────┘    └──────────────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                                                 │
                                                                 ▼
                                              🔍 TOOL EXECUTION (if needed)
                                                                 │
                                    ┌────────────────────────────┼────────────────────────────┐
                                    │                            │                            │
                                    ▼                            ▼                            ▼
                          ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
                          │ GOOGLE SEARCH   │        │   WIKIPEDIA     │        │     ARXIV      │
                          │                 │        │                 │        │                 │
                          │ • Web Results   │        │ • Encyclopedia  │        │ • Research      │
                          │ • Real-time     │        │ • Facts & Info  │        │ • Papers        │
                          │ • Current Data  │        │ • Historical    │        │ • Academic      │
                          └─────────────────┘        └─────────────────┘        └─────────────────┘
                                    │                            │                            │
                                    └────────────────────────────┼────────────────────────────┘
                                                                 │
                                                                 ▼
                                                    📊 AGGREGATED INFORMATION
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RESPONSE GENERATION                                   │
│                                                                                 │
│  ┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐   │
│  │  TEXT RESPONSE  │    │   TEXT PROCESSING    │    │    PHONEME GEN      │   │
│  │                 │───▶│                      │───▶│                     │   │
│  │ • Natural Lang  │    │ • G2P Conversion     │    │ • Misaki Engine     │   │
│  │ • Conversational│    │ • eSpeak Fallback    │    │ • English Phonemes  │   │
│  │ • 1-2 Sentences │    │ • British=False      │    │ • Max Length: 500   │   │
│  └─────────────────┘    └──────────────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         NEURAL VOICE SYNTHESIS                                 │
│                                                                                 │
│  ┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐   │
│  │  KOKORO-82M     │    │   VOICE PROFILES     │    │   AUDIO OUTPUT      │   │
│  │                 │───▶│                      │───▶│                     │   │
│  │ • ONNX Model    │    │ • af_heart (warm)    │    │ • 16kHz Audio       │   │
│  │ • 82M Params    │    │ • af_sky (clear)     │    │ • Natural Speech    │   │
│  │ • High Quality  │    │ • af_bella (dynamic) │    │ • Speed: 1.2x       │   │
│  └─────────────────┘    └──────────────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                                                 │
                                                                 ▼
                                                    🔊 SPEAKER OUTPUT
                                                                 │
                                                                 ▼
                                                      👂 USER HEARS RESPONSE

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PERFORMANCE METRICS                               │
│                                                                                 │
│  Speech Recognition: ~200-500ms  │  LLM Processing: ~1-3s  │  TTS: ~100-300ms  │
│  Total Latency: ~1.3-3.8s       │  Memory Usage: ~4-6GB   │  Concurrent: 2x    │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                KEY FEATURES                                    │
│                                                                                 │
│ 🎙️ Continuous Listening  │ 🧠 Smart Tool Selection │ 🗣️ Natural Voice Output   │
│ ⚡ Real-time Processing  │ 🌐 Web-Connected Intel  │ 🔧 Extensible Architecture │
│ 🎯 Voice Activity Detect │ 📚 Multi-source Search  │ ⚙️ Configurable Settings   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🌟 Key Features

- **🎙️ Real-time Speech Processing**: Advanced speech recognition using Whisper large-v1 with Silero VAD for accurate voice activity detection
- **🧠 Intelligent Reasoning**: Powered by Llama 3.1 8B through the Agno agent framework for sophisticated AI responses
- **🌐 Web-Connected Intelligence**: Integrated web search capabilities (Google Search, Wikipedia, ArXiv) for up-to-date information
- **🗣️ Natural Voice Synthesis**: High-quality speech generation using Kokoro-82M ONNX for human-like voice output
- **⚡ Low-Latency Pipeline**: Optimized audio processing for real-time conversational experience
- **🔧 Extensible Architecture**: Modular tool system allowing easy integration of new capabilities

## 📋 Detailed Application Overview

The Vocal Agent is a specialized speech therapy assistant designed to help users improve their fluency by analyzing and correcting stuttering patterns in real-time. Here's a comprehensive breakdown of how the application works:

### 🔄 Application Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VOCAL AGENT WORKFLOW                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

    🎤 USER SPEAKS (Dysfluent Speech)
         │
         ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   AUDIO CAPTURE     │    │   SPEECH-TO-TEXT     │    │   TEXT ANALYSIS     │
│                     │───▶│   TRANSCRIPTION      │───▶│   & CORRECTION      │
│ • Microphone Input  │    │ • Whisper Model      │    │ • Llama 3.1 8B     │
│ • 16kHz Sampling    │    │ • Multi-language     │    │ • Fluency Rewrite   │
│ • Real-time Stream  │    │ • CUDA Acceleration  │    │ • Pattern Analysis  │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                                                 │
                                                                 ▼
                                                    📝 "Um I don't know what to say Actually yeah."
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AI PROCESSING ENGINE                               │
│                                                                                 │
│  ┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐   │
│  │   FLUENCY       │    │   STUTTER ANALYSIS   │    │   THERAPEUTIC       │   │
│  │   CORRECTION    │───▶│                      │───▶│   SUGGESTIONS       │   │
│  │ • Remove fillers │    │ • Identify patterns │    │ • Exercises         │   │
│  │ • Smooth speech │    │ • Repetition/Blocks  │    │ • Prevention tips    │   │
│  │ • Natural flow  │    │ • Prolongation       │    │ • Improvement strat. │   │
│  └─────────────────┘    └──────────────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                                                 │
                                                                 ▼
                                              🔊 AUDIO RESPONSE GENERATION
                                                                 │
                                    ┌────────────────────────────┼────────────────────────────┐
                                    │                            │                            │
                                    ▼                            ▼                            ▼
                          ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
                          │ FLUENT TEXT     │        │ STUTTER ANALYSIS │        │ THERAPEUTIC     │
                          │ RESPONSE        │        │ REPORT           │        │ RECOMMENDATIONS │
                          │                 │        │                  │        │                 │
                          │ • "I don't know│        │ • Repetition: 3x  │        │ • Practice slow │
                          │   what to say." │        │ • Prolongation: 2x│        │ • Breathing     │
                          │                 │        │ • Interjection: 1x │        │ • exercises     │
                          └─────────────────┘        └─────────────────┘        └─────────────────┘
                                    │                            │                            │
                                    └────────────────────────────┼────────────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         NEURAL VOICE SYNTHESIS                                 │
│                                                                                 │
│  ┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐   │
│  │  KOKORO ONNX    │    │   VOICE PROFILES     │    │   AUDIO OUTPUT      │   │
│  │                 │───▶│                      │───▶│                     │   │
│  │ • 82M Params    │    │ • af_heart (warm)    │    │ • 16kHz Audio       │   │
│  │ • High Quality  │    │ • af_sky (clear)     │    │ • Natural Speech    │   │
│  │ • English TTS   │    │ • af_bella (dynamic) │    │ • Speed: 1.0x       │   │
│  └─────────────────┘    └──────────────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                                                 │
                                                                 ▼
                                                    🔊 SPEAKER OUTPUT
                                                                 │
                                                                 ▼
                                                      👂 USER HEARS CORRECTED RESPONSE

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PERFORMANCE METRICS                               │
│                                                                                 │
│  Speech Recognition: ~200-500ms  │  AI Processing: ~1-3s  │  TTS: ~100-300ms    │
│  Total Latency: ~1.3-3.8s       │  Memory Usage: ~4-6GB   │  Concurrent: 2x    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 🤖 Models and Their Tasks

| Model/Component | Primary Task | Details |
|----------------|--------------|---------|
| **Faster Whisper** | Speech-to-Text Transcription | Converts audio input to text. Supports multiple languages (English, Hindi, Telugu). Uses medium model for accuracy. |
| **Llama 3.1 8B (via Ollama)** | Text Analysis & Correction | Processes transcribed text to:<br>• Remove stuttering and dysfluencies<br>• Analyze speech patterns (repetitions, prolongations, blocks)<br>• Provide therapeutic suggestions and exercises<br>• Generate fluent, natural responses |
| **Kokoro ONNX** | Text-to-Speech Synthesis | Converts corrected text responses into natural-sounding speech audio. Supports English voices with adjustable speed and style. |
| **Agno Framework** | Agent Orchestration | Manages the AI reasoning pipeline, tool integration, and response generation. |
| **Therapy Knowledge Base** | Grounded Suggestions | A YAML file containing evidence-based therapeutic techniques that guides the LLM's suggestions, preventing hallucination. |
| **WebSocket (FastAPI)** | Real-time Communication | Handles bidirectional communication between web interface and backend processing. |

### 📝 Step-by-Step Procedure

1. **Audio Input Capture**
   - User speaks into microphone or uploads audio file
   - Audio is captured at 16kHz sample rate
   - Supports real-time streaming or file upload

2. **Speech Recognition**
   - Audio is processed by Faster Whisper model
   - Transcribed into text with language detection
   - Raw transcription includes all speech patterns (stuttering, fillers, etc.)

3. **AI Analysis & Correction**
   - Transcribed text is sent to Llama 3.1 8B agent
   - Agent processes text using specialized instructions and a **Therapy Knowledge Base**
   - Identifies and analyzes stuttering patterns
   - Rewrites text into fluent, natural speech

4. **Response Generation**
   - Agent generates three-part response:
     - **Corrected Text**: Fluent version of user's input
     - **Analysis**: Clinical breakdown of observed dysfluencies
     - **Suggestions**: Therapeutic recommendations and exercises

5. **Audio Synthesis**
   - Corrected text is converted to speech using Kokoro ONNX
   - Audio generated with selected voice profile and settings
   - Saved as WAV file in `generated_audios` folder for playback

6. **User Feedback**
   - Web interface displays transcription, analysis, and suggestions
   - Audio response plays automatically
   - User can adjust settings and try again

### 🎯 Use Case Example

**Input**: "Um I don't know what to say Actually yeah. Um this weekend I'm going to my cousin's birthday party."

**Processing**:

- **Whisper**: Transcribes the dysfluent speech
- **Llama Agent**:
  - Corrected: "I don't know what to say. This weekend I'm going to my cousin's birthday party."
  - Analysis: "Repetition: 3 instances ('Um'), Interjection: 1 instance ('Actually yeah')"
  - Suggestions: "Practice slow speech techniques, use breathing exercises before speaking"

**Output**: Natural audio playback of the corrected response with accompanying analysis.

## 🏗️ Architecture Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Speech Recognition** | Whisper (large-v1) + Silero VAD | Convert speech to text with voice activity detection |
| **Language Model** | Llama 3.1 8B via Ollama | Natural language understanding and generation |
| **Text-to-Speech** | Kokoro-82M ONNX | Convert text responses to natural speech |
| **Agent Framework** | Agno LLM Agent | Tool orchestration and reasoning capabilities |
| **Web Integration** | Custom API connectors | Real-time information retrieval |

## 📋 Prerequisites

- **Python**: Version 3.9 or higher
- **Ollama**: Local LLM server ([Installation Guide](https://ollama.com/))
- **System Audio**: Microphone and speakers/headphones
- **Operating System**: macOS, Linux, or Windows

## 🚀 Quick Start

### 1. Install Ollama

**macOS:**

```bash
# Download from https://ollama.com/download/mac
# Or install via Homebrew
brew install ollama
```

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**

```bash
# Download installer from https://ollama.com/download/windows
```

### 2. Clone and Setup

```bash
git clone https://github.com/danieladdisonorg/Vocal-Agent.git
cd Vocal-Agent
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip3 install -r requirements.txt
pip3 install --no-deps kokoro-onnx==0.4.7
```

### 4. Install System Dependencies

**Linux:**

```bash
sudo apt-get install espeak-ng
```

**macOS:**

```bash
brew install espeak-ng
```

**Windows:**

1. Download eSpeak NG from [releases page](https://github.com/espeak-ng/espeak-ng/releases)
2. Install the `.msi` package (e.g., `espeak-ng-20191129-b702b03-x64.msi`)

### 5. Download AI Models

**Language Model:**

```bash
ollama pull llama3.1:8b
```

**Voice Models:**
Download the following files and place them in the project root directory:

- [`kokoro-v1.0.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)
- [`voices-v1.0.bin`](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)

## 🎯 Usage

### Starting the Application

1. **Start Ollama service:**

```bash
ollama serve
```

1. **Initialize the model (in a separate terminal):**

```bash
ollama run llama3.1:8b
```

1. **Launch Vocal Agent:**

```bash
python3 main.py
```

### Interaction Flow

```
🎤 Listening... Press Ctrl+C to exit
🔴 Speak now - Recording started
⏹️ Recording stopped

📝 Transcribed: "What's the weather like in Tokyo today?"
🔧 LLM Tool calls...
🤖 Response: "Let me check the current weather in Tokyo for you..."
🔊 [Audio response plays]
```

## ⚙️ Configuration

Customize the application behavior by modifying settings in `main.py`:

```python
# Audio Processing Configuration
SAMPLE_RATE = 16000          # Audio sample rate (Hz)

# Performance Tier (choose one: "low_latency", "balanced", "high_quality")
ACTIVE_TIER = "balanced"

# Voice Synthesis Settings
SPEED = 1.2                  # Speech rate multiplier
VOICE_PROFILE = "af_heart"   # Voice character selection
```

### Performance Tiers

You can balance performance and quality by changing the `ACTIVE_TIER` variable.

| Tier | Whisper Model | Compute Type | Use Case |
|--------------|---------------|--------------|------------------------------------------|
| `low_latency`| `base` | `int8` | Fastest response, suitable for older hardware. |
| `balanced` | `medium` | `int8` | Good balance of speed and accuracy. (Default) |
| `high_quality`| `large-v3` | `float16` | Highest accuracy, requires powerful GPU. |

### Therapy Knowledge Base

The agent's therapeutic suggestions are grounded in an evidence-based knowledge base located in `therapy_knowledge_base.yml`. You can extend or modify this file to customize the advice provided by the agent.

# Performance Settings
MAX_THREADS = 2              # Parallel processing threads
```

### Available Voice Profiles

- `af_heart` - Warm, friendly tone
- `af_sky` - Clear, professional tone
- `af_bella` - Expressive, dynamic tone
- Additional profiles available in `voices-v1.0.bin`

## 📁 Project Structure

```
Vocal-Agent/
├── main.py                 # Core application entry point
├── agent_client.py         # LLM agent integration layer
├── kokoro-v1.0.onnx       # Neural TTS model
├── voices-v1.0.bin        # Voice profile database
├── requirements.txt       # Python dependencies
├── vocal_agent_mac.sh     # macOS setup automation script
├── demo.png              # Application demonstration
├── LICENSE               # MIT license
└── README.md            # Project documentation
```

## 🛠️ Development

### Extending Functionality

Add new tools to the agent by integrating [Agno Toolkits](https://docs.agno.com/tools/toolkits/toolkits):

```python
from agno import Agent
from agno.tools import WebSearchTool, WikipediaSearchTool

# Add custom tools
agent = Agent(
    tools=[WebSearchTool(), WikipediaSearchTool(), YourCustomTool()],
    model="llama3.1:8b"
)
```

### Performance Optimization

- **GPU Acceleration**: Enable CUDA for faster model inference
- **Model Selection**: Choose smaller models for faster response times
- **Audio Buffer Tuning**: Adjust buffer sizes for your hardware

## 🔧 Troubleshooting

### Common Issues

**Ollama Connection Error:**

```bash
# Ensure Ollama is running
ollama serve
# Verify model is available
ollama list
```

**Audio Device Issues:**

- Check microphone permissions
- Verify audio device selection in system settings
- Test with `python3 -c "import sounddevice; print(sounddevice.query_devices())"`

**Model Download Failures:**

- Ensure stable internet connection
- Verify sufficient disk space (models require ~8GB)
- Check Ollama service status

## 📊 Performance Metrics

- **Speech Recognition Latency**: ~200-500ms
- **LLM Response Time**: ~1-3 seconds (depending on query complexity)
- **Text-to-Speech Generation**: ~100-300ms
- **Memory Usage**: ~4-6GB (with Llama 3.1 8B)

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)** - Speech-to-text with VAD integration
- **[Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx)** - Efficient neural text-to-speech
- **[Agno](https://docs.agno.com/introduction)** - Powerful agent framework
- **[Ollama](https://ollama.ai/)** - Local LLM serving platform
- **[Weebo](https://github.com/amanvirparhar/weebo)** - Project inspiration

## 📞 Support

- **Documentation**: [Project Wiki](https://github.com/danieladdisonorg/Vocal-Agent/wiki)
- **Issues**: [GitHub Issues](https://github.com/danieladdisonorg/Vocal-Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/danieladdisonorg/Vocal-Agent/discussions)
