# main.py — FastAPI application entry point
#
# Auth design:
#  - Each user gets a random session_id stored in an HttpOnly cookie.
#  - Google OAuth tokens live in _sessions[session_id] — NEVER sent to the browser.
#  - All calendar operations resolve credentials from the session, not a shared store.
#  - Sessions are fully isolated; one user cannot access another's calendar.

import os
import secrets
from datetime import date, timedelta

from dotenv import load_dotenv
from fastapi import Cookie, FastAPI
from fastapi.responses import JSONResponse, RedirectResponse

from calendar_tool import (
    credentials_from_flow,
    get_free_slots,
    get_google_flow,
    refresh_if_expired,
)

load_dotenv()

app = FastAPI(title="Smart Scheduler")

# ── Server-side session store ──────────────────────────────────────────────────
# Maps session_id -> google.oauth2.credentials.Credentials
# In production with multiple workers, replace with Redis.
_sessions: dict = {}

# Maps OAuth state token -> session_id (prevents CSRF on the callback)
_oauth_states: dict[str, str] = {}

COOKIE_NAME = "ss_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


def _get_valid_credentials(session_id: str | None):
    """Return refreshed credentials for session_id, or None if not authenticated."""
    if not session_id or session_id not in _sessions:
        return None
    try:
        creds = refresh_if_expired(_sessions[session_id])
    except Exception:
        # Refresh failed (revoked token, etc.) — force re-auth
        _sessions.pop(session_id, None)
        return None
    if not creds.valid:
        _sessions.pop(session_id, None)
        return None
    _sessions[session_id] = creds  # persist refreshed token
    return creds


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "smart-scheduler"}


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
        slots = get_free_slots(creds, tomorrow, duration, tz_name=tz)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)

    return {
        "date": tomorrow,
        "timezone": tz,
        "duration_minutes": duration,
        "available_slots": slots,
        "count": len(slots),
    }
