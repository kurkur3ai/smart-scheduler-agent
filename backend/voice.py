# voice.py — STT + TTS via Groq

import logging
import os
import re
import time

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
            language="en",
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
    # Strip markdown formatting so TTS doesn't speak "asterisk" etc.
    clean = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)  # bold/italic
    clean = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', clean)      # underscore bold/italic
    clean = re.sub(r'`+.*?`+', '', clean)                      # inline code
    clean = re.sub(r'^#{1,6}\s*', '', clean, flags=re.MULTILINE)  # headings
    # Replace time-range dashes/en-dashes with 'to' so TTS reads naturally
    # Handles both '9 AM–10 AM' and '9:30 AM–10:30 AM'
    clean = re.sub(
        r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM))\s*[\u2013\u2014\-]\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM))',
        r'\1 to \2', clean
    )
    clean = clean.strip()
    # Orpheus has a 200-character input limit
    truncated = clean[:200]
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
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
            last_exc = exc
            if attempt < 2:
                log.warning("[voice] tts transient error (attempt %d/3), retrying: %s", attempt + 1, exc)
                time.sleep(0.5 * (attempt + 1))
                continue
            break
    log.error("[voice] tts error: %s", last_exc)
    raise RuntimeError(f"Speech synthesis failed: {last_exc}") from last_exc


async def speak_text_iter(text: str):
    """
    Async generator that streams WAV bytes from Orpheus without buffering the
    full response in the backend.  Use with FastAPI StreamingResponse.

    The same text-cleaning logic as speak_text() is applied.  Orpheus returns
    standard PCM WAV (16-bit little-endian) so the frontend can progressively
    schedule chunks via the Web Audio API.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    # Identical cleaning pipeline as speak_text()
    clean = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    clean = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', clean)
    clean = re.sub(r'`+.*?`+', '', clean)
    clean = re.sub(r'^#{1,6}\s*', '', clean, flags=re.MULTILINE)
    clean = re.sub(
        r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM))\s*[\u2013\u2014\-]\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM))',
        r'\1 to \2', clean
    )
    clean = clean.strip()[:200]

    import httpx as _httpx  # already imported at module level; alias avoids shadowing
    async with _httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://api.groq.com/openai/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "canopylabs/orpheus-v1-english",
                "voice": "troy",
                "input": clean,
                "response_format": "wav",
            },
            timeout=30.0,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise RuntimeError(
                    f"Speech synthesis failed: {response.status_code} {body.decode(errors='replace')}"
                )
            log.info("[voice] streaming TTS for %d chars", len(clean))
            async for chunk in response.aiter_bytes(chunk_size=4096):
                yield chunk

