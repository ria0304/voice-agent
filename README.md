# 🎙️ VoiceAgent — Local Voice-Controlled AI Agent

> **Mem0 AI/ML Generative AI Developer Intern Assignment**

A local AI agent that accepts audio input, classifies intent, executes tools, and displays the full pipeline in a polished Streamlit UI.

---

## 🏗️ Architecture

```
Audio/Text Input
      │
      ▼
┌─────────────┐
│  STT Module │  ← whisper-local / Groq / OpenAI Whisper
└──────┬──────┘
       │ Transcribed Text
       ▼
┌──────────────────┐
│ Intent Classifier│  ← LLM (Ollama / Anthropic / OpenAI)
└────────┬─────────┘
         │ Intents + Params (supports compound commands)
         ▼
┌──────────────────┐
│  Tool Executor   │  ← sandboxed to ./output/
│  ─────────────── │
│  create_file     │
│  write_code      │
│  summarize_text  │
│  general_chat    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Streamlit UI    │  ← live pipeline display + session history
└──────────────────┘
```

---

## ✨ Features

| Feature | Status |
|---|---|
| Microphone / file upload audio input | ✅ |
| Local Whisper STT (CPU/GPU) | ✅ |
| Groq & OpenAI Whisper (cloud fallback) | ✅ |
| Ollama LLM (local, any model) | ✅ |
| Anthropic Claude & OpenAI (cloud fallback) | ✅ |
| `create_file` intent | ✅ |
| `write_code` intent (multi-language) | ✅ |
| `summarize_text` intent | ✅ |
| `general_chat` intent with memory | ✅ |
| **Compound commands** (e.g. "summarize and save to file") | ✅ Bonus |
| **Human-in-the-loop** confirmation before file ops | ✅ Bonus |
| **Graceful degradation** (keyword fallback on LLM error) | ✅ Bonus |
| **Session memory** (chat context across turns) | ✅ Bonus |
| Full pipeline displayed in UI | ✅ |
| Sandboxed output (`./output/` only) | ✅ |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/voice-agent
cd voice-agent
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure credentials
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. (Optional) Start Ollama for local LLM
```bash
# Install from https://ollama.com
ollama serve
ollama pull llama3   # or mistral, phi3, etc.
```

### 6. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## ⚙️ Configuration

All settings are in the sidebar:

| Setting | Options |
|---|---|
| STT Provider | `whisper-local`, `groq`, `openai` |
| LLM Provider | `ollama`, `anthropic`, `openai` |
| Ollama Model | any model name pulled via `ollama pull` |
| Confirm before file ops | toggle |

### Recommended combos

| Use case | STT | LLM |
|---|---|---|
| Fully local (no API keys) | `whisper-local` | `ollama` |
| Fast cloud | `groq` | `anthropic` |
| OpenAI only | `openai` | `openai` |

---

## 🔊 Hardware Notes

### Local Whisper
- `openai-whisper` with the `base` model runs comfortably on **CPU** (~5–10s per clip).
- For faster inference on GPU: uncomment `faster-whisper` in `requirements.txt` and update `stt.py`.

### Why cloud STT is offered
If your machine lacks sufficient RAM/GPU for real-time inference (e.g. a Raspberry Pi or low-end laptop), Groq's Whisper endpoint provides near-instant transcription for free with generous rate limits. This is documented as a valid alternative in the assignment spec.

---

## 📂 Output Safety

All generated files are written exclusively to the **`./output/`** directory. Path traversal is stripped from filenames in `tools.py:_safe_path()`. The output folder is git-ignored.

---

## 💡 Example Commands

```
"Create a Python file with a binary search function"
"Summarize: The quick brown fox jumps over the lazy dog repeatedly."
"What is recursion?"
"Write a JavaScript fetch wrapper with retry logic and save it as fetcher.js"
"Summarize this meeting note and save it to summary.txt"
```

---

## 🧠 Models Used

| Component | Default | Alternative |
|---|---|---|
| STT | openai-whisper base (local) | Groq whisper-large-v3 |
| Intent + Code | Ollama llama3 | Claude claude-opus-4-5 |
| Summarization | Same LLM | — |

---

## 📄 License
MIT
"# voice-agent" 
