# main.py — FastAPI application entry point
#
# Auth design:
#  - Each user gets a random session_id stored in an HttpOnly cookie.
#  - Google OAuth tokens live in _sessions[session_id] — NEVER sent to the browser.
#  - All calendar operations resolve credentials from the session, not a shared store.
#  - Sessions are fully isolated; one user cannot access another's calendar.

import json
import logging
import os
import secrets
import time
import uvicorn
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from google.oauth2.credentials import Credentials
from pydantic import BaseModel

from agent import run_agent
from calendar_tool import (
    SCOPES,
    credentials_from_flow,
    get_availability,
    get_google_flow,
    refresh_if_expired,
)

load_dotenv()  # works when CWD == smart-scheduler/
# Also try parent dir in case CWD == smart-scheduler/backend/
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Logging ────────────────────────────────────────────────────────────────────
_is_dev = os.getenv("ENVIRONMENT", "development") == "development"
logging.basicConfig(
    level=logging.DEBUG if _is_dev else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")
# Silence noisy third-party loggers even in dev
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("googleapiclient").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("python_multipart").setLevel(logging.WARNING)

app = FastAPI(title="Smart Scheduler")


@app.on_event("startup")
async def _startup() -> None:
    from calendar_tool import warmup
    warmup()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    ms = (time.monotonic() - start) * 1000
    log.debug("%s %s → %d (%.0fms)", request.method, request.url.path, response.status_code, ms)
    return response

# ── Dev-mode credential persistence ────────────────────────────────────────────
# In dev, serialize credentials to .credentials/<session_id>.json so they
# survive --reload restarts. In production this is never used.
_CREDS_DIR = Path(__file__).parent.parent / ".credentials"


def _save_creds_to_disk(session_id: str, creds: Credentials) -> None:
    if not _is_dev:
        return
    _CREDS_DIR.mkdir(exist_ok=True)
    (_CREDS_DIR / f"{session_id}.json").write_text(creds.to_json())


def _load_creds_from_disk(session_id: str) -> Credentials | None:
    if not _is_dev:
        return None
    path = _CREDS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return Credentials.from_authorized_user_info(json.loads(path.read_text()), SCOPES)
    except Exception:
        return None


# ── Server-side session store ──────────────────────────────────────────────────
# Maps session_id -> google.oauth2.credentials.Credentials
# In production with multiple workers, replace with Redis.
_sessions: dict = {}

# Maps OAuth state token -> session_id (prevents CSRF on the callback)
_oauth_states: dict[str, str] = {}

# Maps session_id -> conversation history (list of {role, content} dicts)
# Enhanced with entity extraction + expiry in Component 5.
_conversations: dict[str, list] = {}

# Per-session freebusy cache: {session_id: {"date:sh:eh": {"result": ..., "expires": float}}}
# Avoids redundant Google API calls when availability was already checked this session.
_slot_cache: dict[str, dict] = {}

COOKIE_NAME = "ss_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days

FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"


def _get_valid_credentials(session_id: str | None):
    """Return refreshed credentials for session_id, or None if not authenticated."""
    if not session_id:
        return None
    # Hot path: already in memory
    if session_id not in _sessions:
        # Cold path: try to restore from disk (dev only)
        creds = _load_creds_from_disk(session_id)
        if creds:
            _sessions[session_id] = creds
            log.debug("[auth] restored session from disk: %s...", session_id[:8])
        else:
            return None
    try:
        creds = refresh_if_expired(_sessions[session_id])
    except Exception:
        _sessions.pop(session_id, None)
        return None
    if not creds.valid:
        _sessions.pop(session_id, None)
        return None
    _sessions[session_id] = creds
    return creds


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "smart-scheduler"}


@app.get("/config")
async def config():
    """
    Tells the frontend which UI mode to render.
    dev  → text chat box (no voice)
    prod → voice UI (Component 6)
    Never exposes secrets — only the mode string.
    """
    return {"mode": os.getenv("ENVIRONMENT", "development")}


@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND, media_type="text/html")


# ── OAuth ──────────────────────────────────────────────────────────────────────

@app.get("/auth/login")
async def auth_login(ss_session: str = Cookie(default=None)):
    """
    Start the Google OAuth2 flow.
    Reuses an existing session_id if the browser already has one,
    otherwise mints a new one. The session_id cookie is set (or refreshed)
    here so it is available when the callback arrives.
    """
    session_id = ss_session or secrets.token_urlsafe(32)

    flow = get_google_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",  # always return a refresh_token
    )
    _oauth_states[state] = session_id

    response = RedirectResponse(auth_url)
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_id,
        httponly=True,       # JS cannot read this cookie
        samesite="lax",
        max_age=COOKIE_MAX_AGE,
        secure=os.getenv("ENVIRONMENT", "development") == "production",
    )
    return response


@app.get("/auth/callback")
async def auth_callback(code: str, state: str):
    """
    Google redirects here after the user grants consent.
    Exchange the code for tokens and bind them to the user's session.
    The token is stored server-side only — never returned to the client.
    """
    if state not in _oauth_states:
        return JSONResponse(
            {"error": "Invalid OAuth state parameter. Possible CSRF attempt."},
            status_code=400,
        )
    session_id = _oauth_states.pop(state)

    try:
        flow = get_google_flow()
        creds = credentials_from_flow(flow, code)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Token exchange failed: {exc}"}, status_code=500
        )

    _sessions[session_id] = creds
    _save_creds_to_disk(session_id, creds)
    log.info("[auth] session authenticated: %s...", session_id[:8])

    response = RedirectResponse("/")
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=COOKIE_MAX_AGE,
        secure=os.getenv("ENVIRONMENT", "development") == "production",
    )
    return response


@app.get("/auth/status")
async def auth_status(ss_session: str = Cookie(default=None)):
    """
    Called by the frontend on page load to decide whether to show
    the Login with Google Calendar screen or the main UI.
    Returns only a boolean — no token data ever leaves the server.
    """
    return {"authenticated": _get_valid_credentials(ss_session) is not None}


@app.post("/auth/logout")
async def auth_logout(ss_session: str = Cookie(default=None)):
    """Revoke the server-side session. The cookie is cleared on the client."""
    if ss_session:
        _sessions.pop(ss_session, None)
        # Clean up disk file in dev
        creds_file = _CREDS_DIR / f"{ss_session}.json"
        if creds_file.exists():
            creds_file.unlink()
        log.info("[auth] session logged out: %s...", ss_session[:8])
    response = JSONResponse({"status": "logged out"})
    response.delete_cookie(COOKIE_NAME)
    return response


# ── Test route ─────────────────────────────────────────────────────────────────

@app.get("/test-calendar")
async def test_calendar(
    ss_session: str = Cookie(default=None),
    duration: int = 30,
    tz: str = "UTC",
):
    """
    Returns available slots for **tomorrow** for the authenticated user.

    Query params:
        duration  — meeting length in minutes (default 30)
        tz        — IANA timezone, e.g. America%2FNew_York (default UTC)

    Requires the ss_session cookie set by /auth/login -> /auth/callback.
    """
    creds = _get_valid_credentials(ss_session)
    if not creds:
        return JSONResponse(
            {
                "error": "Not authenticated.",
                "fix": "Open http://localhost:8000/auth/login in your browser.",
            },
            status_code=401,
        )

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    try:
        slots = get_availability(creds, tomorrow, tz_name=tz)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)

    return {
        "date": tomorrow,
        "timezone": tz,
        "duration_minutes": duration,
        "available_slots": slots,
        "count": len(slots),
    }


# ── Chat ───────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    timezone: str = "UTC"


@app.post("/chat")
async def chat(
    body: ChatRequest,
    ss_session: str = Cookie(default=None),
):
    """
    Send a text message to the scheduling agent and receive a text reply.

    The agent may call get_free_slots or book_slot transparently — the
    caller only ever sees the final natural-language response.

    Requires the ss_session cookie (set by /auth/login → /auth/callback).
    """
    creds = _get_valid_credentials(ss_session)
    if not creds:
        return JSONResponse(
            {
                "error": "Not authenticated.",
                "fix": "Visit /auth/login first.",
            },
            status_code=401,
        )

    history = _conversations.get(ss_session, [])
    slot_cache = _slot_cache.setdefault(ss_session, {})

    t_chat = time.monotonic()
    try:
        reply, updated_history, agent_timing = run_agent(
            user_message=body.message,
            history=history,
            credentials=creds,
            user_timezone=body.timezone,
            slot_cache=slot_cache,
        )
    except RuntimeError as exc:
        log.error("[chat] agent error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=502)

    chat_total_ms = round((time.monotonic() - t_chat) * 1000)
    log.info("[chat] session=%s... turns=%d total=%dms", ss_session[:8], len(updated_history) // 2, chat_total_ms)

    # Log per-step breakdown in dev
    if _is_dev:
        for gc in agent_timing.get("groq_calls", []):
            log.info("  [timing] LLM call iter=%d: %dms", gc["iter"], gc["ms"])
        for tc in agent_timing.get("tools", []):
            log.info("  [timing] tool %-25s: %dms", tc["name"], tc["ms"])
        log.info("  [timing] agent total: %dms", agent_timing.get("total_ms", 0))

    _conversations[ss_session] = updated_history
    resp: dict = {"reply": reply}
    if _is_dev:
        resp["timing"] = agent_timing
    return resp


# ── Voice ──────────────────────────────────────────────────────────────────────

@app.post("/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
    ss_session: str = Cookie(default=None),
):
    """
    Transcribe an audio recording to text using Groq Whisper.
    Accepts any browser-recorded audio format (webm, mp4, wav).
    """
    creds = _get_valid_credentials(ss_session)
    if not creds:
        return JSONResponse({"error": "Not authenticated."}, status_code=401)

    audio_bytes = await audio.read()
    filename = audio.filename or "audio.webm"

    from voice import transcribe_audio
    t_stt = time.monotonic()
    try:
        text = transcribe_audio(audio_bytes, filename)
    except RuntimeError as exc:
        log.error("[voice] transcribe error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=502)
    stt_ms = round((time.monotonic() - t_stt) * 1000)
    if _is_dev:
        log.info("  [timing] STT transcribe: %dms (%d bytes → %d chars)", stt_ms, len(audio_bytes), len(text))

    resp: dict = {"text": text}
    if _is_dev:
        resp["timing_ms"] = stt_ms
    return resp


class SynthesizeRequest(BaseModel):
    text: str


@app.post("/voice/synthesize")
async def voice_synthesize(
    body: SynthesizeRequest,
    ss_session: str = Cookie(default=None),
):
    """
    Convert agent reply text to MP3 audio using Groq TTS (PlayAI).
    Returns audio/mpeg bytes.
    """
    creds = _get_valid_credentials(ss_session)
    if not creds:
        return JSONResponse({"error": "Not authenticated."}, status_code=401)

    from voice import speak_text
    t_tts = time.monotonic()
    try:
        audio_bytes = speak_text(body.text)
    except RuntimeError as exc:
        log.error("[voice] synthesize error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=502)
    tts_ms = round((time.monotonic() - t_tts) * 1000)
    if _is_dev:
        log.info("  [timing] TTS synthesize: %dms (%d chars → %d bytes)", tts_ms, len(body.text), len(audio_bytes))

    headers = {"X-Timing-Ms": str(tts_ms)} if _is_dev else {}
    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)



if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)