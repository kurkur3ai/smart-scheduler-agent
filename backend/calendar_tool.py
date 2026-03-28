# calendar_tool.py — Google Calendar API integration
#
# Design: stateless helpers — credentials are ALWAYS passed in from the caller.
# The server-side session store lives in main.py. Tokens never touch the filesystem
# and are never sent to the browser.

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]


# ── OAuth helpers ──────────────────────────────────────────────────────────────

def _client_config() -> dict:
    """Build OAuth client config from env vars — never from a file on disk."""
    return {
        "web": {
            "client_id": os.environ["GOOGLE_CLIENT_ID"],
            "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [os.environ["GOOGLE_REDIRECT_URI"]],
        }
    }


def get_google_flow() -> Flow:
    return Flow.from_client_config(
        _client_config(),
        scopes=SCOPES,
        redirect_uri=os.environ["GOOGLE_REDIRECT_URI"],
    )


def credentials_from_flow(flow: Flow, code: str) -> Credentials:
    """Exchange the authorisation code for tokens. Returns Credentials object."""
    flow.fetch_token(code=code)
    return flow.credentials


def refresh_if_expired(creds: Credentials) -> Credentials:
    """Refresh the access token in-place if it has expired. Returns the same object."""
    if creds.expired and creds.refresh_token:
        creds.refresh(GoogleAuthRequest())
    return creds


# ── Calendar operations — all accept credentials as first argument ─────────────

def get_free_slots(
    creds: Credentials,
    date: str,
    duration_minutes: int,
    tz_name: str = "UTC",
    work_start_hour: int = 8,
    work_end_hour: int = 18,
) -> list[dict]:
    """
    Return available meeting slots on a given date for the authenticated user.

    Args:
        creds:            Google OAuth2 Credentials for the logged-in user.
        date:             YYYY-MM-DD string.
        duration_minutes: Desired meeting length in minutes.
        tz_name:          IANA timezone name (e.g. "America/New_York").
        work_start_hour:  First bookable hour of the day (24h). Default 8.
        work_end_hour:    Meetings must end by this hour. Default 18.

    Returns:
        List of {"start": <ISO str>, "end": <ISO str>} dicts, chronological order.

    Raises:
        RuntimeError: If the Google Calendar API call fails.
    """
    creds = refresh_if_expired(creds)

    tz = ZoneInfo(tz_name)
    day = datetime.strptime(date, "%Y-%m-%d")
    day_start = day.replace(hour=work_start_hour, minute=0, second=0, tzinfo=tz)
    day_end = day.replace(hour=work_end_hour, minute=0, second=0, tzinfo=tz)

    service = build("calendar", "v3", credentials=creds)
    body = {
        "timeMin": day_start.isoformat(),
        "timeMax": day_end.isoformat(),
        "items": [{"id": "primary"}],
    }
    try:
        result = service.freebusy().query(body=body).execute()
    except Exception as exc:
        raise RuntimeError(f"Google Calendar API error: {exc}") from exc

    # Parse busy periods — Google returns RFC 3339 strings
    busy: list[tuple[datetime, datetime]] = []
    for period in result["calendars"]["primary"]["busy"]:
        start = datetime.fromisoformat(
            period["start"].replace("Z", "+00:00")
        ).astimezone(tz)
        end = datetime.fromisoformat(
            period["end"].replace("Z", "+00:00")
        ).astimezone(tz)
        busy.append((start, end))
    busy.sort(key=lambda x: x[0])

    # Walk the working day, collecting gaps large enough for the requested duration
    free_slots: list[dict] = []
    slot_delta = timedelta(minutes=duration_minutes)
    cursor = day_start

    for busy_start, busy_end in busy:
        while cursor + slot_delta <= busy_start:
            free_slots.append(
                {
                    "start": cursor.isoformat(),
                    "end": (cursor + slot_delta).isoformat(),
                }
            )
            cursor += slot_delta
        if cursor < busy_end:
            cursor = busy_end

    # Slots after the last busy block
    while cursor + slot_delta <= day_end:
        free_slots.append(
            {
                "start": cursor.isoformat(),
                "end": (cursor + slot_delta).isoformat(),
            }
        )
        cursor += slot_delta

    return free_slots


def book_slot(
    creds: Credentials,
    start: str,
    end: str,
    summary: str,
    attendees: list[str] | None = None,
    description: str = "",
) -> dict:
    """
    Create a calendar event for the logged-in user.

    Args:
        creds:      Google OAuth2 Credentials.
        start:      ISO 8601 start datetime string (with timezone offset).
        end:        ISO 8601 end datetime string.
        summary:    Event title.
        attendees:  Optional list of attendee email addresses.
        description: Optional event description.

    Returns:
        The created event resource dict from the Google Calendar API.

    Raises:
        RuntimeError: If the API call fails.
    """
    creds = refresh_if_expired(creds)
    service = build("calendar", "v3", credentials=creds)

    event = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start},
        "end": {"dateTime": end},
    }
    if attendees:
        event["attendees"] = [{"email": e} for e in attendees]

    try:
        created = service.events().insert(calendarId="primary", body=event).execute()
    except Exception as exc:
        raise RuntimeError(f"Failed to create event: {exc}") from exc

    return created
