# Local English Practice Coach

A local English-practice app built with Streamlit, Ollama (`llama3.2`) and an OpenAI-compatible local TTS Studio API.

## Features

- English conversation practice at B1/B2 level
- Practice focus and role-play scenario selection
- Gentle grammar and phrasing corrections
- Follow-up questions to keep the conversation going
- Text-to-speech using the `Annike` or `Anton` voice
- Conversation history stored in Streamlit session state

## Requirements

- Python 3.10+
- Ollama running locally with `llama3.2` downloaded
- Your local TTS Studio running on port `7849`
- A TTS Studio API token

## 1. Start Ollama

```bash
ollama pull llama3.2
ollama serve
```

If Ollama is already running as a service, you do not need to run `ollama serve` again.

Check the model:

```bash
ollama list
```

## 2. Configure the TTS API

In TTS Studio, copy the API token from the Connection screen. Create a local `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and replace `PASTE_YOUR_LOCAL_TTS_TOKEN_HERE` with your token.

The app sends the token using the OpenAI SDK as a Bearer token. Do not commit `.env` to Git.

The app assumes the TTS service is reachable at `http://127.0.0.1:7849/v1` and that it accepts the voice names `Annike` and `Anton`.

## 3. Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally:

`http://localhost:8501`

## How to use

1. Choose your English level, practice focus, scenario and coach voice.
2. Enter a message in the chat box.
3. Read the coach's answer and suggested correction.
4. Click **Speak latest reply** to hear it, or enable automatic speech in the sidebar.

## Important networking note

The TTS Studio screenshot says the service listens on `127.0.0.1` only. Therefore, run this Streamlit app on the same computer as TTS Studio. If you run Streamlit inside Docker or on another machine, `127.0.0.1` will refer to that container or machine, not your desktop.

## Troubleshooting

### Ollama connection error

Check that Ollama is running and that this works:

```bash
curl http://127.0.0.1:11434/api/tags
```

### TTS authentication error

Check the token in `.env`. Create a new token in TTS Studio if the old one was revoked.

### TTS voice or model error

The app uses the exact model name from the example (`phonon-40m-multilingual`) and sends `Annike` or `Anton` as the `voice` value. If your TTS Studio API expects different voice IDs, use the exact IDs shown in its voice/reference list.

### Audio format

The app displays generated audio as MP3. If your server returns WAV instead, change `format="audio/mpeg"` in `app.py` to `format="audio/wav"`.
