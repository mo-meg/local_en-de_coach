
import os
import tempfile
from functools import lru_cache

import numpy as np
import requests
import soundfile as sf
from openai import OpenAI


def get_ollama_models(base_url):
    """Retrieve locally installed Ollama models."""
    try:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]

        response = requests.get(
            f"{root}/api/tags",
            timeout=5
        )
        response.raise_for_status()

        return [
            model["name"]
            for model in response.json().get("models", [])
        ]
    except Exception:
        return []


@lru_cache(maxsize=2)
def load_parakeet_model(model_name, cpu_threads):
    """Load and cache Parakeet V3."""
    import torch
    import nemo.collections.asr as nemo_asr

    torch.set_num_threads(int(cpu_threads))

    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=model_name
    )

    model.eval()
    model = model.to("cpu")

    return model


def transcribe_parakeet(
    audio_bytes,
    model_name="nvidia/parakeet-tdt-0.6b-v3",
    cpu_threads=4
):
    """Transcribe recorded WAV audio using Parakeet."""

    model = load_parakeet_model(
        model_name,
        int(cpu_threads)
    )

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as temp:
        temp.write(audio_bytes)
        audio_path = temp.name

    try:
        signal, sample_rate = sf.read(audio_path)

        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        if sample_rate != 16000:
            raise ValueError(
                f"Expected 16000 Hz audio, got {sample_rate} Hz."
            )

        if len(signal) == 0:
            return ""

        # NeMo accepts a list of audio file paths.
        result = model.transcribe(
            [audio_path],
            batch_size=1
        )

        if not result:
            return ""

        first = result[0]

        if isinstance(first, str):
            return first.strip()

        # NeMo versions may return a Hypothesis object.
        if hasattr(first, "text"):
            return first.text.strip()

        return str(first).strip()

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def synthesize_speech(
    text,
    base_url,
    api_key,
    model,
    voice
):
    """Generate audio through the local OpenAI-compatible TTS API."""

    if not api_key:
        raise ValueError("TTS API token is missing.")

    client = OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        timeout=180
    )

    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )

    return response.content
