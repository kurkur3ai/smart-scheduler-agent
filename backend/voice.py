# voice.py — STT + TTS via Groq

import logging
import os
import re

import httpx
from groq import Groq

log = logging.getLogger("voice")

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"), timeout=30.0)
    return _client


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Transcribe audio to text using Groq Whisper.

    Args:
        audio_bytes: Raw audio data (webm, mp4, wav, etc.).
        filename:    Filename hint with extension so Groq can detect the format.

    Returns:
        Transcribed text string.

    Raises:
        RuntimeError: If the Groq API call fails.
    """
    try:
        client = _get_client()
        # Use default (json) response format — returns an object with .text attribute
        # response_format="text" behaves inconsistently across SDK versions
        result = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
        )
        text = result.text if hasattr(result, "text") else str(result)
        log.info("[voice] transcribed %d bytes → %d chars", len(audio_bytes), len(text))
        return text.strip()
    except Exception as exc:
        log.error("[voice] transcribe error: %s", exc)
        raise RuntimeError(f"Transcription failed: {exc}") from exc


def speak_text(text: str) -> bytes:
    """Convert text to WAV audio bytes using Groq TTS (PlayAI).

    Uses the REST API directly because Groq SDK v0.13.x does not expose
    client.audio.speech — that attribute was added in a later SDK release.

    Returns:
        WAV audio bytes.

    Raises:
        RuntimeError: If the API call fails.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    try:
        # Strip markdown formatting so TTS doesn't speak "asterisk" etc.
        clean = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)  # bold/italic
        clean = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', clean)      # underscore bold/italic
        clean = re.sub(r'`+.*?`+', '', clean)                      # inline code
        clean = re.sub(r'^#{1,6}\s*', '', clean, flags=re.MULTILINE)  # headings
        clean = clean.strip()
        # Orpheus has a 200-character input limit
        truncated = clean[:200]
        response = httpx.post(
            "https://api.groq.com/openai/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "canopylabs/orpheus-v1-english",
                "voice": "troy",
                "input": truncated,
                "response_format": "wav",
            },
            timeout=30.0,
        )
        response.raise_for_status()
        audio_bytes = response.content
        log.info("[voice] synthesized %d chars → %d bytes", len(text), len(audio_bytes))
        return audio_bytes
    except httpx.HTTPStatusError as exc:
        log.error("[voice] tts HTTP error %d: %s", exc.response.status_code, exc.response.text)
        raise RuntimeError(f"Speech synthesis failed: {exc.response.status_code} {exc.response.text}") from exc
    except Exception as exc:
        log.error("[voice] tts error: %s", exc)
        raise RuntimeError(f"Speech synthesis failed: {exc}") from exc

