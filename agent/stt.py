

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def transcribe_audio(audio_file, provider: str = "whisper-local") -> str:
    """
    Transcribe audio to text.

    Parameters
    ----------
    audio_file : file-like or path
        Uploaded audio buffer from Streamlit or a file path string.
    provider : str
        One of "whisper-local", "groq", "openai".

    Returns
    -------
    str
        Transcribed text.
    """
    if provider == "whisper-local":
        return _transcribe_local(audio_file)
    elif provider == "groq":
        return _transcribe_groq(audio_file)
    elif provider == "openai":
        return _transcribe_openai(audio_file)
    else:
        raise ValueError(f"Unknown STT provider: {provider}")


# ── Local Whisper ──────────────────────────────────────────────────────────────

def _transcribe_local(audio_file) -> str:
    try:
        import whisper  # openai-whisper package
    except ImportError:
        raise ImportError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        )

    model = whisper.load_model("base")  # base is a good speed/accuracy tradeoff

    # Write to a temp file so whisper can read it
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.read() if hasattr(audio_file, "read") else open(audio_file, "rb").read())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path)
        return result["text"].strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Groq Whisper API ───────────────────────────────────────────────────────────

def _transcribe_groq(audio_file) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment / .env")

    client = Groq(api_key=api_key)

    audio_bytes = audio_file.read() if hasattr(audio_file, "read") else open(audio_file, "rb").read()
    filename = getattr(audio_file, "name", "audio.wav")

    transcription = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
        response_format="text",
    )
    return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()


# ── OpenAI Whisper API ─────────────────────────────────────────────────────────

def _transcribe_openai(audio_file) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment / .env")

    client = OpenAI(api_key=api_key)

    audio_bytes = audio_file.read() if hasattr(audio_file, "read") else open(audio_file, "rb").read()
    filename = getattr(audio_file, "name", "audio.wav")

    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        return transcript.text.strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
