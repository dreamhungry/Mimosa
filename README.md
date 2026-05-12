# 🌿 Mimosa

Mimosa is an evolving virtual companion that senses and responds to human emotion through multimodal signals, building a personalized and adaptive relationship over time.

Inspired by the mimosa plant, which reacts to external stimuli, Mimosa explores how AI systems can exhibit sensitivity, personality, and long-term relational dynamics with humans.

## ✨ Features

- **Voice & Text Chat** - Talk via text or microphone with real-time responses
- **Multi-LLM Support** - Compatible with OpenAI, DeepSeek, Gemini, Ollama
- **Live2D Avatar** - Animated character with emotion-driven expressions
- **Text-to-Speech** - AI responses are spoken aloud via Edge TTS
- **Speech Recognition** - Offline ASR powered by Sherpa ONNX
- **Personality System** - Big Five-based configurable personality with self-evolution
- **Conversation Memory** - Persistent chat history across sessions

## 🏗️ Architecture

```
Frontend (Web)                    Backend (FastAPI)
┌─────────────────┐              ┌────────────────────────────┐
│ Live2D Canvas   │◄── WS ──────►│ WebSocket Handler          │
│ Chat UI         │              │   ├── Perception Layer     │
│ Audio I/O       │              │   │   ├── VAD (Silero)     │
└─────────────────┘              │   │   └── ASR (Sherpa)     │
                                 │   ├── Core Layer           │
                                 │   │   ├── LLM (Multi)      │
                                 │   │   └── Memory           │
                                 │   └── Response Layer       │
                                 │       ├── TTS (Edge)       │
                                 │       └── Emotion Extract  │
                                 └────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Mimosa.git
cd Mimosa

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Configuration

Edit `conf.yaml` to set up your LLM provider:

```yaml
llm:
  provider: "openai_compatible"
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key-here"
```

**Examples for different providers:**

```yaml
# DeepSeek
llm:
  provider: "openai_compatible"
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com/v1"
  api_key: "your-deepseek-key"

# Ollama (local)
llm:
  provider: "openai_compatible"
  model: "llama3"
  base_url: "http://localhost:11434/v1"
  api_key: ""

# Gemini
llm:
  provider: "openai_compatible"
  model: "gemini-pro"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
  api_key: "your-gemini-key"
```

### Run

```bash
python run_server.py
```

Then open http://localhost:8000 in your browser.

## 📁 Project Structure

```
Mimosa/
├── pyproject.toml           # Dependencies
├── conf.yaml                # Configuration
├── model_dict.json          # Live2D model registry
├── run_server.py            # Entry point
├── prompts/                 # Persona prompts
├── character/               # Personality config & state
│   ├── personality.yaml.example
│   └── personality_state.yaml.example
├── live2d-models/           # Live2D model files
├── src/mimosa/              # Backend source
│   ├── config.py            # Config management
│   ├── server.py            # FastAPI app
│   ├── routes.py            # WebSocket routes
│   ├── api_routes.py        # REST API routes
│   ├── websocket_handler.py # Message handling
│   ├── service_context.py   # Service container
│   ├── llm/                 # LLM module
│   ├── asr/                 # Speech recognition
│   ├── tts/                 # Speech synthesis
│   ├── vad/                 # Voice activity detection
│   ├── memory/              # Chat history & long-term memory
│   ├── personality/         # Personality system
│   ├── live2d/              # Live2D model management
│   └── conversation/        # Conversation orchestration
└── frontend/                # Web frontend
    ├── index.html
    ├── style.css
    └── js/
```

## 🎨 Supported Models

| Provider | Example Models | base_url |
|----------|---------------|----------|
| OpenAI | gpt-4o-mini, gpt-4o | https://api.openai.com/v1 |
| DeepSeek | deepseek-chat | https://api.deepseek.com/v1 |
| Gemini | gemini-pro | https://generativelanguage.googleapis.com/v1beta/openai |
| Ollama | llama3, qwen2 | http://localhost:11434/v1 |

## 🧠 Personality System

Mimosa features a personality system based on the [Big Five personality model](https://en.wikipedia.org/wiki/Big_Five_personality_traits) that can be configured and evolves over time through conversations.

### Configuration

Copy the example files to create your personality configuration:

```bash
cp character/personality.yaml.example character/personality.yaml
```

Edit `character/personality.yaml` to customize the baseline personality:

```yaml
big_five:
  openness: 70             # 0~100: Curiosity, imagination
  conscientiousness: 50    # 0~100: Organization, dependability
  extraversion: 60         # 0~100: Sociability, assertiveness
  agreeableness: 80        # 0~100: Cooperation, empathy
  neuroticism: 30          # 0~100: Emotional sensitivity

style:
  humor_style: "gentle"    # gentle / witty / sarcastic / dry
  speech_formality: 30     # 0=casual, 100=formal
  quirks:
    - "uses plant metaphors occasionally"

evolution:
  enabled: true
  interval: 100            # Evolve every N conversation turns
  max_delta: 5             # Max change per dimension per cycle
  min_value: 10
  max_value: 100
```

### Self-Evolution

When enabled, personality evolves through two triggers:

1. **Periodic** — Every N conversation turns (configurable via `evolution.interval`)
2. **Session end** — When a WebSocket client disconnects

The evolution engine uses LLM reflection to analyze conversation patterns and propose bounded adjustments (±5 per dimension, clamped to [10, 100]). Evolution history is logged to `character/evolution_history.jsonl`.

To reset personality to baseline:

```bash
# Via API
curl -X POST http://localhost:8000/api/personality/reset

# Or manually: delete the state file and restart
rm character/personality_state.yaml
```

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/personality` | Get current personality state (Big Five + style) |
| POST | `/api/personality/reset` | Reset personality to baseline |

**Example response** for `GET /api/personality`:

```json
{
  "status": "ok",
  "data": {
    "big_five": {
      "openness": 73,
      "conscientiousness": 50,
      "extraversion": 62,
      "agreeableness": 80,
      "neuroticism": 28
    },
    "style": {
      "humor_style": "gentle",
      "speech_formality": 30,
      "quirks": ["uses plant metaphors occasionally"]
    }
  }
}
```

## 📄 License

MIT License
