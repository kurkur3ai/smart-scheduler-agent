# calendar_tool.py — Google Calendar API integration
#
# Design: stateless helpers — credentials are ALWAYS passed in from the caller.
# The server-side session store lives in main.py. Tokens never touch the filesystem
# and are never sent to the browser.

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Load .env from the project root (works whether CWD is smart-scheduler/ or backend/)
load_dotenv()
load_dotenv(Path(__file__).parent.parent / ".env")
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.discovery_cache.base import Cache

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]


class _MemoryCache(Cache):
    """In-process cache for the Google API discovery document.
    Avoids an extra HTTP round-trip on every build() call."""
    _store: dict = {}
    def get(self, url):        return self._store.get(url)
    def set(self, url, content): self._store[url] = content


_DISCOVERY_CACHE = _MemoryCache()

log = logging.getLogger("calendar")

# Cache built service objects so the underlying HTTP connection is reused.
# Key = access token string (changes on refresh → triggers a rebuild automatically).
_service_cache: dict[str, object] = {}


def _get_service(creds: Credentials, version: str = "v3"):
    key = creds.token
    if key not in _service_cache:
        _service_cache.clear()  # old tokens are useless — don't accumulate
        _service_cache[key] = build(
            "calendar", version, credentials=creds,
            cache=_DISCOVERY_CACHE, static_discovery=True,
        )
    return _service_cache[key]


def warmup() -> None:
    """Pre-resolve DNS and warm TCP/TLS to googleapis.com in a background thread."""
    import threading
    import urllib.request

    def _do() -> None:
        try:
            t = time.perf_counter()
            urllib.request.urlopen("https://www.googleapis.com/", timeout=5)
        except Exception:
            pass  # 404/403 is fine — we only care about DNS + TLS
        finally:
            log.debug("[timing] warmup (DNS+TLS)  %.0fms", (time.perf_counter() - t) * 1000)

    threading.Thread(target=_do, daemon=True).start()


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

def _fetch_day_data(
    creds: Credentials,
    date: str,
    tz_name: str,
    busy_cache: dict | None,
) -> dict:
    """Fetch or return cached events for an entire calendar day.
    Uses events().list() to return both busy periods and event titles in one call.
    Cache key = date:tz_name. Returns {"busy": [...], "events": [...], "expires": float}"""
    bkey = f"{date}:{tz_name}"
    if busy_cache is not None and bkey in busy_cache:
        entry = busy_cache[bkey]
        if time.time() < entry["expires"]:
            log.debug("[tool] day CACHE HIT %s", bkey)
            return entry

    creds = refresh_if_expired(creds)
    tz = ZoneInfo(tz_name)
    day = datetime.strptime(date, "%Y-%m-%d")
    day_start = day.replace(hour=0, minute=0, second=0, tzinfo=tz)
    day_end = (day + timedelta(days=1)).replace(hour=0, minute=0, second=0, tzinfo=tz)

    _t = time.perf_counter()
    service = _get_service(creds)
    log.debug("[timing] calendar_build  %.0fms", (time.perf_counter() - _t) * 1000)

    try:
        _t = time.perf_counter()
        result = service.events().list(
            calendarId="primary",
            timeMin=day_start.isoformat(),
            timeMax=day_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        log.debug("[timing] events_list  %.0fms", (time.perf_counter() - _t) * 1000)
    except Exception as exc:
        raise RuntimeError(f"Google Calendar API error: {exc}") from exc

    busy: list[dict] = []
    events: list[dict] = []
    for item in result.get("items", []):
        start_str = item.get("start", {}).get("dateTime")
        end_str = item.get("end", {}).get("dateTime")
        if not start_str or not end_str:
            continue  # skip all-day events
        title = item.get("summary", "(No title)")
        busy.append({"start": start_str, "end": end_str})
        events.append({"title": title, "start": start_str, "end": end_str})

    data: dict = {"busy": busy, "events": events, "expires": time.time() + 300}
    if busy_cache is not None:
        busy_cache[bkey] = data
    return data


def compute_free_windows(
    busy_raw: list,
    window_start: datetime,
    window_end: datetime,
    tz: ZoneInfo,
    min_minutes: int = 15,
) -> list[dict]:
    """Compute merged free windows within window_start..window_end from raw busy data."""
    busy: list[tuple] = []
    for p in busy_raw:
        s = datetime.fromisoformat(p["start"].replace("Z", "+00:00")).astimezone(tz)
        e = datetime.fromisoformat(p["end"].replace("Z", "+00:00")).astimezone(tz)
        if s < window_end and e > window_start:
            busy.append((max(s, window_start), min(e, window_end)))
    busy.sort(key=lambda x: x[0])

    min_delta = timedelta(minutes=min_minutes)
    windows: list[dict] = []
    cursor = window_start
    for bs, be in busy:
        if cursor + min_delta <= bs:
            windows.append({"from": cursor.isoformat(), "to": bs.isoformat()})
        if cursor < be:
            cursor = be
    if cursor + min_delta <= window_end:
        windows.append({"from": cursor.isoformat(), "to": window_end.isoformat()})
    return windows


def get_availability(
    creds: Credentials,
    date: str,
    tz_name: str = "UTC",
    busy_cache: dict | None = None,
) -> list[dict]:
    """Return free windows for the full day (from current time if today)."""
    tz = ZoneInfo(tz_name)
    day = datetime.strptime(date, "%Y-%m-%d")
    day_start = day.replace(hour=0, minute=0, second=0, tzinfo=tz)
    day_end = (day + timedelta(days=1)).replace(hour=0, minute=0, second=0, tzinfo=tz)

    now_local = datetime.now(tz)
    if day.date() == now_local.date():
        mins = now_local.minute
        round_up = timedelta(minutes=(-mins % 15))
        now_rounded = now_local.replace(second=0, microsecond=0) + round_up
        if now_rounded > day_start:
            day_start = now_rounded

    if day_start >= day_end:
        return []

    busy_raw = _fetch_day_data(creds, date, tz_name, busy_cache)["busy"]
    return compute_free_windows(busy_raw, day_start, day_end, tz, 15)


def check_slot(
    creds: Credentials,
    date: str,
    start_iso: str,
    end_iso: str,
    tz_name: str = "UTC",
    busy_cache: dict | None = None,
) -> dict:
    """Check if a specific ISO 8601 time slot is free.
    Returns {"available": True, "start": ..., "end": ...}
    or      {"available": False, "conflict": {"start": ..., "end": ...}}
    """
    tz = ZoneInfo(tz_name)
    slot_start = datetime.fromisoformat(start_iso).astimezone(tz)
    slot_end = datetime.fromisoformat(end_iso).astimezone(tz)

    busy_raw = _fetch_day_data(creds, date, tz_name, busy_cache)["busy"]
    for p in busy_raw:
        bs = datetime.fromisoformat(p["start"].replace("Z", "+00:00")).astimezone(tz)
        be = datetime.fromisoformat(p["end"].replace("Z", "+00:00")).astimezone(tz)
        if slot_start < be and slot_end > bs:
            return {"available": False, "conflict": {"start": bs.isoformat(), "end": be.isoformat()}}

    return {"available": True, "start": slot_start.isoformat(), "end": slot_end.isoformat()}


def find_next_slot(
    creds: Credentials,
    date: str,
    duration_minutes: int,
    tz_name: str = "UTC",
    not_before_time: str | None = None,
    not_after_time: str | None = None,
    busy_cache: dict | None = None,
) -> dict:
    """Find the first available slot of duration_minutes on date.

    Args:
        not_before_time: Optional "HH:MM" earliest start in user's timezone.
        not_after_time:  Optional "HH:MM" latest end time. Defaults to 22:00.

    Returns:
        {"available": True, "start_iso": ..., "end_iso": ..., "date": ..., "duration_minutes": ...}
        or {"available": False, "date": ..., "duration_minutes": ...}
    """
    tz = ZoneInfo(tz_name)
    day = datetime.strptime(date, "%Y-%m-%d")
    day_start = day.replace(hour=0, minute=0, second=0, tzinfo=tz)
    day_end = (day + timedelta(days=1)).replace(hour=0, minute=0, second=0, tzinfo=tz)

    # Apply not_before constraint
    if not_before_time:
        h, m = map(int, not_before_time.split(":"))
        not_before_dt = day.replace(hour=h, minute=m, second=0, tzinfo=tz)
        if not_before_dt > day_start:
            day_start = not_before_dt

    # Apply not_after constraint (default 22:00)
    cap_time = not_after_time or "22:00"
    ch, cm = map(int, cap_time.split(":"))
    cap_dt = day.replace(hour=ch, minute=cm, second=0, tzinfo=tz)
    if cap_dt < day_end:
        day_end = cap_dt

    # If today, advance past current time (rounded up to next 15 min)
    now_local = datetime.now(tz)
    if day.date() == now_local.date():
        mins = now_local.minute
        round_up = timedelta(minutes=(-mins % 15))
        now_rounded = now_local.replace(second=0, microsecond=0) + round_up
        if now_rounded > day_start:
            day_start = now_rounded

    if day_start >= day_end:
        return {"available": False, "date": date, "duration_minutes": duration_minutes}

    busy_raw = _fetch_day_data(creds, date, tz_name, busy_cache)["busy"]
    duration_delta = timedelta(minutes=duration_minutes)

    windows = compute_free_windows(busy_raw, day_start, day_end, tz, duration_minutes)
    for w in windows:
        w_start = datetime.fromisoformat(w["from"])
        slot_end = w_start + duration_delta
        w_end = datetime.fromisoformat(w["to"])
        if slot_end <= w_end:
            return {
                "available": True,
                "start_iso": w_start.isoformat(),
                "end_iso": slot_end.isoformat(),
                "date": date,
                "duration_minutes": duration_minutes,
            }

    return {"available": False, "date": date, "duration_minutes": duration_minutes}


def get_events_for_day(
    creds: Credentials,
    date: str,
    tz_name: str = "UTC",
    busy_cache: dict | None = None,
) -> list[dict]:
    """Return events with titles for a day. Uses cache if already fetched — no extra API call."""
    data = _fetch_day_data(creds, date, tz_name, busy_cache)
    return data.get("events", [])


def search_event(
    creds: Credentials,
    query: str,
    start_date: str,
    end_date: str,
    tz_name: str = "UTC",
) -> dict:
    """Search for calendar events by keyword across a date range."""
    creds = refresh_if_expired(creds)
    tz = ZoneInfo(tz_name)
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(hour=0, minute=0, second=0, tzinfo=tz)
    end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, tzinfo=tz
    )
    service = _get_service(creds)
    try:
        _t = time.perf_counter()
        result = service.events().list(
            calendarId="primary",
            q=query,
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=10,
        ).execute()
        log.debug("[timing] events_search  %.0fms", (time.perf_counter() - _t) * 1000)
    except Exception as exc:
        raise RuntimeError(f"Google Calendar API error: {exc}") from exc

    events = []
    for item in result.get("items", []):
        start_str = item.get("start", {}).get("dateTime") or item.get("start", {}).get("date")
        end_str = item.get("end", {}).get("dateTime") or item.get("end", {}).get("date")
        events.append({
            "title": item.get("summary", "(No title)"),
            "start": start_str,
            "end": end_str,
            "date": (start_str or "")[:10],
        })
    return {"events": events, "count": len(events), "query": query}


def find_slot_in_range(
    creds: Credentials,
    start_date: str,
    end_date: str,
    duration_minutes: int,
    tz_name: str = "UTC",
    not_before_time: str | None = None,
    not_after_time: str | None = None,
    exclude_days: list[str] | None = None,
    busy_cache: dict | None = None,
) -> dict:
    """Find the first available slot across a date range, fetching days in parallel.

    Args:
        exclude_days:    Lowercase day names to skip, e.g. ["wednesday", "saturday"].
        not_before_time: Earliest start time each day as "HH:MM". Defaults to "08:00".
        not_after_time:  Latest end time each day as "HH:MM". Defaults to "22:00".
    """
    tz = ZoneInfo(tz_name)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    excluded = {d.lower() for d in (exclude_days or [])}
    duration_delta = timedelta(minutes=duration_minutes)

    dates: list[str] = []
    d = start
    while d <= end:
        if d.strftime("%A").lower() not in excluded:
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    if not dates:
        return {"available": False, "start_date": start_date, "end_date": end_date, "duration_minutes": duration_minutes}

    # Refresh once before spawning threads
    creds = refresh_if_expired(creds)

    def _fetch(date_str: str) -> tuple:
        try:
            return date_str, _fetch_day_data(creds, date_str, tz_name, None)
        except Exception as exc:
            log.error("[tool] find_slot_in_range fetch error %s: %s", date_str, exc)
            return date_str, None

    with ThreadPoolExecutor(max_workers=min(len(dates), 5)) as ex:
        fetched: dict = dict(ex.map(_fetch, dates))

    # Populate shared cache from parallel results (single-threaded, safe)
    if busy_cache is not None:
        for date_str, data in fetched.items():
            if data:
                busy_cache[f"{date_str}:{tz_name}"] = data

    nb_h, nb_m = map(int, (not_before_time or "08:00").split(":"))
    na_h, na_m = map(int, (not_after_time or "22:00").split(":"))

    for date_str in dates:
        data = fetched.get(date_str)
        if not data:
            continue
        day = datetime.strptime(date_str, "%Y-%m-%d")
        day_start = day.replace(hour=nb_h, minute=nb_m, second=0, tzinfo=tz)
        day_end = day.replace(hour=na_h, minute=na_m, second=0, tzinfo=tz)

        now_local = datetime.now(tz)
        if day.date() == now_local.date():
            mins = now_local.minute
            now_rounded = now_local.replace(second=0, microsecond=0) + timedelta(minutes=(-mins % 15))
            if now_rounded > day_start:
                day_start = now_rounded

        if day_start >= day_end:
            continue

        windows = compute_free_windows(data["busy"], day_start, day_end, tz, duration_minutes)
        for w in windows:
            w_start = datetime.fromisoformat(w["from"])
            slot_end = w_start + duration_delta
            if slot_end <= datetime.fromisoformat(w["to"]):
                return {
                    "available": True,
                    "start_iso": w_start.isoformat(),
                    "end_iso": slot_end.isoformat(),
                    "date": date_str,
                    "duration_minutes": duration_minutes,
                }

    return {"available": False, "start_date": start_date, "end_date": end_date, "duration_minutes": duration_minutes}


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
    _t = time.perf_counter()
    creds = refresh_if_expired(creds)
    log.debug("[timing] token_refresh  %.0fms", (time.perf_counter() - _t) * 1000)

    _t = time.perf_counter()
    service = _get_service(creds)
    log.debug("[timing] calendar_build  %.0fms", (time.perf_counter() - _t) * 1000)

    event = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start},
        "end": {"dateTime": end},
    }
    if attendees:
        event["attendees"] = [{"email": e} for e in attendees]

    try:
        _t = time.perf_counter()
        created = service.events().insert(calendarId="primary", body=event).execute()
        log.debug("[timing] events_insert  %.0fms", (time.perf_counter() - _t) * 1000)
    except Exception as exc:
        raise RuntimeError(f"Failed to create event: {exc}") from exc

    return created
