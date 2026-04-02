# agent.py â€” 2-step scheduling pipeline (token-optimized)
#
# Architecture:                                         ~Input tokens
#   Step 0: _fast_intent()   â€” 0 LLM tokens (set lookup)        0
#   Step 1: extract_intent() â€” minimal LLM call               ~150
#   Step 2: _route()         â€” Python router, 0 LLM tokens       0
#   Step 3: _fallback_reply()â€” only for unknown intent         ~80
#
# Typical turn: 1 LLM call Ã— ~150 tokens
# vs old tool-calling: 2-3 LLM calls Ã— ~2400 tokens each.

import json
import logging
import os
import time
from datetime import datetime, timedelta, date as _date
from zoneinfo import ZoneInfo

from groq import Groq, RateLimitError
from google.oauth2.credentials import Credentials

from calendar_tool import (
    get_availability, check_slot, book_slot, find_next_slot,
    get_events_for_day, search_event, find_slot_in_range,
)
from intent import extract_intent

log = logging.getLogger("agent")

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"), timeout=20.0)
MODEL_PRIMARY  = "llama-3.3-70b-versatile"
MODEL_FALLBACK = "llama-3.1-8b-instant"



def _fmt_time(dt: datetime) -> str:
    hour = int(dt.strftime("%I"))  # strips leading zero, cross-platform
    return f"{hour}:{dt.strftime('%M %p')}"


def _fmt_date(date_str: str) -> str:
    d = datetime.strptime(date_str, "%Y-%m-%d")
    day = str(int(d.strftime("%d")))  # strip leading zero, cross-platform
    return d.strftime(f"%A, %B {day}")


def _fmt_windows(windows: list) -> str:
    parts = []
    for w in windows:
        try:
            s = datetime.fromisoformat(w["from"])
            e = datetime.fromisoformat(w["to"])
            parts.append(f"{_fmt_time(s)}\u2013{_fmt_time(e)}")
        except Exception:
            pass
    return ", ".join(parts) if parts else "none"


# â”€â”€ ISO helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_PERIOD_TIMES = {
    "morning": "09:00", "afternoon": "13:00", "evening": "18:00",
    "night": "19:00", "noon": "12:00", "midnight": "00:00",
    "tonight": "19:00", "lunchtime": "12:00", "lunch": "12:00",
}


def _build_iso(date_str: str, time_str: str, tz_name: str) -> str:
    tz = ZoneInfo(tz_name)
    # Resolve period words → concrete HH:MM before parsing
    resolved = _PERIOD_TIMES.get(time_str.strip().lower(), time_str)
    dt = datetime.strptime(f"{date_str} {resolved}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
    return dt.isoformat()


def _today_str(tz_name: str) -> str:
    return datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")


def _intent_to_context(intent: dict) -> str:
    """Compact string injected as 'Prev:' in the next intent extraction call."""
    parts = []
    for key in ("title", "date", "duration_minutes", "time"):
        val = intent.get(key)
        if val:
            parts.append(f"{key}={val}")
    if intent.get("date_range"):
        dr = intent["date_range"]
        parts.append(f"range={dr.get('start')} to {dr.get('end')}")
    # Carry constraints so mid-conversation changes (e.g. new duration) keep morning/evening preference
    c = intent.get("constraints") or {}
    if c.get("not_before"):
        parts.append(f"not_before={c['not_before']}")
    if c.get("not_after"):
        parts.append(f"not_after={c['not_after']}")
    if c.get("exclude_days"):
        parts.append(f"exclude_days={','.join(c['exclude_days'])}")
    return ",".join(parts)


# â”€â”€ Step 0: zero-token fast intent detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_CONFIRM_WORDS = {
    "yes", "yeah", "yep", "sure", "ok", "okay", "go ahead",
    "confirm", "book it", "do it", "yup", "absolutely", "schedule it",
}
_CANCEL_WORDS = {
    "no", "nope", "cancel", "stop", "never mind", "nevermind",
    "forget it", "dont", "don't", "nah",
}


def _fast_intent(user_message: str) -> str | None:
    """Detect obvious confirm/cancel without an LLM call. Returns action str or None."""
    msg = user_message.strip().lower().rstrip(".")
    if msg in _CONFIRM_WORDS:
        return "confirm"
    if msg in _CANCEL_WORDS:
        return "cancel"
    return None


def _last_weekday_of_month(tz_name: str) -> str:
    """Return YYYY-MM-DD of the last Mon–Fri of the current month."""
    tz = ZoneInfo(tz_name)
    today = datetime.now(tz).date()
    if today.month == 12:
        last_day = _date(today.year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = _date(today.year, today.month + 1, 1) - timedelta(days=1)
    while last_day.weekday() >= 5:  # 5=Sat, 6=Sun
        last_day -= timedelta(days=1)
    return last_day.strftime("%Y-%m-%d")


def _preprocess_message(msg: str, tz_name: str) -> str:
    """Replace known date expressions with concrete dates before calling the LLM."""
    import re as _re
    if _re.search(r"last\s+weekday\s+of\s+(this|the)\s+month", msg, _re.IGNORECASE):
        concrete = _last_weekday_of_month(tz_name)
        msg = _re.sub(
            r"last\s+weekday\s+of\s+(this|the)\s+month",
            concrete, msg, flags=_re.IGNORECASE,
        )
        log.debug("[preprocess] last weekday of month → %s", concrete)
    return msg


# â”€â”€ Step 2: deterministic router (zero LLM tokens) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _route(
    intent: dict,
    creds: Credentials,
    slot_cache: dict,
    tz_name: str,
    today: str,
    timing: dict,
) -> str | None:
    """
    Route a structured intent to calendar operations and return a reply string.
    Returns None only for 'unknown' intent (caller uses NL fallback).
    Zero LLM calls â€” all scheduling logic lives here.
    """
    action = intent.get("action", "unknown")
    log.info("[router] action=%s intent=%s", action, json.dumps(intent)[:200])

    # Normalise LLM action variants → canonical actions
    if action in ("suggest", "schedule", "create", "add"):
        action = "find_slot"

    # Normalise: find_slot with explicit HH:MM time → check slot directly
    # But if time is a period word (morning/afternoon/evening), keep as find_slot with not_before
    time_val = intent.get("time") or ""
    is_period = time_val.strip().lower() in _PERIOD_TIMES if time_val else False

    # Guard: if time matches not_after or not_before, the LLM confused deadline for start time.
    # Move it to the appropriate constraint and route to find_slot instead.
    # Use `or {}` to handle both missing key AND key present with null value from LLM.
    constraints = intent.get("constraints") or {}
    intent["constraints"] = constraints
    if time_val and not is_period:
        not_after_val = constraints.get("not_after")
        not_before_val = constraints.get("not_before")
        if time_val == not_after_val or time_val == not_before_val:
            # LLM duplicated deadline time into `time` field — clear it
            intent["time"] = None
            time_val = ""

    if action == "find_slot" and time_val and not is_period:
        action = "book_explicit"
    elif action == "find_slot" and is_period:
        # Convert period to not_before constraint and clear the time field
        period_key = time_val.strip().lower()
        _period_not_before = {"morning": "08:00", "afternoon": "12:00",
                               "evening": "17:00", "night": "18:00", "tonight": "18:00"}
        if not constraints.get("not_before"):
            constraints["not_before"] = _period_not_before.get(period_key, "08:00")
        intent["time"] = None
        time_val = ""
    # Normalise: book_explicit without time or range → search for a free slot
    if action == "book_explicit" and not intent.get("time") and not intent.get("date_range"):
        action = "find_slot"

    # â”€â”€ CONFIRM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if action == "confirm":
        pending = slot_cache.get("_pending")
        if not pending:
            return "I'm not sure what to confirm \u2014 what would you like to book?"
        _t = time.perf_counter()
        try:
            book_slot(
                creds=creds,
                start=pending["start_iso"],
                end=pending["end_iso"],
                summary=pending["title"],
                description="",
            )
            timing["tools"].append({"name": "book_slot", "ms": round((time.perf_counter() - _t) * 1000)})
            slot_cache.pop("_pending", None)
            return f"Done! **{pending['title']}** is on your calendar for {pending['display']}."
        except Exception as exc:
            log.error("[router] book_slot error: %s", exc)
            return f"Couldn't book it: {exc}"

    # â”€â”€ CANCEL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if action == "cancel":
        slot_cache.pop("_pending", None)
        return "Alright, nothing booked. Let me know if you need anything else."

    # â”€â”€ LIST EVENTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if action == "list_events":
        date = intent.get("date") or today
        date_label = _fmt_date(date)
        _t = time.perf_counter()
        try:
            events = get_events_for_day(creds, date, tz_name, slot_cache)
            timing["tools"].append({"name": "get_events", "ms": round((time.perf_counter() - _t) * 1000)})
        except Exception as exc:
            return f"Couldn't fetch calendar: {exc}"
        if not events:
            return f"You're completely free on {date_label}."
        parts = []
        for e in events:
            try:
                s = datetime.fromisoformat(e["start"])
                et = datetime.fromisoformat(e["end"])
                parts.append(f"{_fmt_time(s)}\u2013{_fmt_time(et)}: {e['title']}")
            except Exception:
                parts.append(e.get("title", "?"))
        return f"On {date_label}: " + "; ".join(parts) + "."

    # â”€â”€ SEARCH EVENT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if action == "search_event":
        query = intent.get("anchor_event") or intent.get("title") or ""
        if not query:
            return "What event are you looking for?"
        start_date = intent.get("date") or today
        end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
        _t = time.perf_counter()
        try:
            result = search_event(creds, query, start_date, end_date, tz_name)
            timing["tools"].append({"name": "search_event", "ms": round((time.perf_counter() - _t) * 1000)})
        except Exception as exc:
            return f"Search failed: {exc}"
        events = result.get("events", [])
        if not events:
            return f"No events found matching '{query}'."
        e = events[0]
        try:
            s = datetime.fromisoformat(e["start"])
            return f"Found '{e['title']}' on {e['date']} at {_fmt_time(s)}."
        except Exception:
            return f"Found: {e.get('title', query)}"

    # â”€â”€ CHECK AVAILABILITY or BOOK EXPLICIT (specific time given) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if action in ("check_availability", "book_explicit"):
        date = intent.get("date")
        if not date:
            return "Which day are you asking about?"
        specific_time = intent.get("time")
        duration = int(intent.get("duration_minutes") or 60)
        title = intent.get("title") or "Meeting"
        date_label = _fmt_date(date)

        if not specific_time:
            # General availability query \u2014 list free windows
            _t = time.perf_counter()
            try:
                windows = get_availability(creds, date, tz_name, slot_cache)
                timing["tools"].append({"name": "get_availability", "ms": round((time.perf_counter() - _t) * 1000)})
            except Exception as exc:
                return f"Couldn't check calendar: {exc}"
            if not windows:
                return f"You're fully booked on {date_label}."
            fitting = [
                w for w in windows
                if (datetime.fromisoformat(w["to"]) - datetime.fromisoformat(w["from"])).seconds >= duration * 60
            ]
            return f"On {date_label} you're free: {_fmt_windows(fitting or windows)}."

        # Specific time — verify availability then set pending confirmation
        try:
            start_iso = _build_iso(date, specific_time, tz_name)
            end_iso = (datetime.fromisoformat(start_iso) + timedelta(minutes=duration)).isoformat()
        except (ValueError, KeyError) as exc:
            return f"Couldn't parse that time: {exc}"

        # Guard: reject slots already in the past
        tz = ZoneInfo(tz_name)
        slot_start_dt = datetime.fromisoformat(start_iso).astimezone(tz)
        if slot_start_dt < datetime.now(tz):
            try:
                windows = get_availability(creds, date, tz_name, slot_cache)
                alts = _fmt_windows(windows[:3])
                past_msg = f"{_fmt_time(slot_start_dt)} has already passed."
                return (f"{past_msg} Still free on {date_label}: {alts}."
                        if alts else f"{past_msg} You're fully booked for the rest of {date_label}.")
            except Exception:
                return f"{_fmt_time(slot_start_dt)} has already passed."

        _t = time.perf_counter()
        try:
            result = check_slot(creds, date, start_iso, end_iso, tz_name, slot_cache)
            timing["tools"].append({"name": "check_slot", "ms": round((time.perf_counter() - _t) * 1000)})
        except Exception as exc:
            return f"Couldn't check that slot: {exc}"
        s_dt = datetime.fromisoformat(start_iso)
        e_dt = datetime.fromisoformat(end_iso)
        if result.get("available"):
            display = f"{date_label} at {_fmt_time(s_dt)}\u2013{_fmt_time(e_dt)}"
            slot_cache["_pending"] = {
                "title": title, "start_iso": start_iso,
                "end_iso": end_iso, "timezone": tz_name, "display": display,
            }
            return f"Booking **{title}** on {display}. Shall I go ahead?"
        else:
            conflict = result.get("conflict", {})
            try:
                cs = datetime.fromisoformat(conflict["start"])
                ce = datetime.fromisoformat(conflict["end"])
                conflict_str = f" (conflict: {_fmt_time(cs)}\u2013{_fmt_time(ce)})"
            except Exception:
                conflict_str = ""
            try:
                windows = get_availability(creds, date, tz_name, slot_cache)
                alts = _fmt_windows(windows[:3])
                return f"That time is taken{conflict_str}. Free slots on {date_label}: {alts}."
            except Exception:
                return f"That time is taken{conflict_str}."

    # â”€â”€ FIND SLOT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if action == "find_slot":
        date = intent.get("date")
        date_range = intent.get("date_range")
        duration = int(intent.get("duration_minutes") or 60)
        title = intent.get("title") or "Meeting"
        constraints = intent.get("constraints") or {}
        not_before = constraints.get("not_before")  # defer default until after anchor resolution
        not_after = constraints.get("not_after")
        exclude_days = constraints.get("exclude_days") or []

        # Resolve anchor event â†’ time constraint (e.g. "after my standup")
        # Resolve anchor event → time constraint or date offset
        # "after my standup"            → not_before = standup_end
        # "day or two after event"      → date_range shifts by anchor_offset_days
        # "1 hour buffer after meeting" → not_before = meeting_end + buffer_minutes
        anchor = intent.get("anchor_event")
        anchor_relation = intent.get("anchor_relation") or "after"
        anchor_offset_days = int(intent.get("anchor_offset_days") or 0)
        buffer_minutes = int(intent.get("buffer_minutes") or 0)
        if anchor:
            anchor_search_start = date or today
            anchor_search_end = (
                datetime.strptime(anchor_search_start, "%Y-%m-%d") + timedelta(days=14)
            ).strftime("%Y-%m-%d")
            _t = time.perf_counter()
            try:
                anc = search_event(creds, anchor, anchor_search_start, anchor_search_end, tz_name)
                timing["tools"].append({"name": "search_event(anchor)", "ms": round((time.perf_counter() - _t) * 1000)})
                anc_events = anc.get("events", [])
                if anc_events:
                    ae = anc_events[0]
                    ae_date = ae["date"]
                    if anchor_offset_days:
                        # e.g. "a day or two after Project Alpha" → shift search date
                        new_start = (
                            datetime.strptime(ae_date, "%Y-%m-%d") + timedelta(days=anchor_offset_days)
                        ).strftime("%Y-%m-%d")
                        new_end = (
                            datetime.strptime(ae_date, "%Y-%m-%d") + timedelta(days=anchor_offset_days + 1)
                        ).strftime("%Y-%m-%d")
                        if not date_range:
                            date_range = {"start": new_start, "end": new_end}
                            date = None
                        log.debug("[router] anchor '%s' +%dd offset → range %s to %s",
                                  anchor, anchor_offset_days, new_start, new_end)
                    elif not (not_before or not_after):
                        # Same-day time constraint
                        if anchor_relation == "before":
                            not_after = datetime.fromisoformat(ae["start"]).strftime("%H:%M")
                        else:
                            end_dt = datetime.fromisoformat(ae["end"])
                            if buffer_minutes:
                                end_dt += timedelta(minutes=buffer_minutes)
                            not_before = end_dt.strftime("%H:%M")
                        log.debug("[router] anchor '%s' → %s=%s (buffer=%dmin)",
                                  anchor,
                                  "not_after" if anchor_relation == "before" else "not_before",
                                  not_after or not_before, buffer_minutes)
            except Exception as exc:
                log.warning("[router] anchor search failed: %s", exc)

        # Apply default working-hours floor AFTER anchor resolution
        # (so anchor end-time isn't overwritten by "08:00" default)
        if not not_before:
            not_before = "08:00"

        if not date and not date_range:
            return "Which day (or date range) should I look for a slot?"

        if date_range:
            start_date = date_range.get("start", today)
            end_date = date_range.get("end", today)
            _t = time.perf_counter()
            try:
                result = find_slot_in_range(
                    creds, start_date, end_date, duration, tz_name,
                    not_before_time=not_before, not_after_time=not_after,
                    exclude_days=exclude_days or None, busy_cache=slot_cache,
                )
                timing["tools"].append({"name": "find_slot_in_range", "ms": round((time.perf_counter() - _t) * 1000)})
            except Exception as exc:
                return f"Couldn't search for slots: {exc}"
        else:
            _t = time.perf_counter()
            try:
                result = find_next_slot(
                    creds, date, duration, tz_name,
                    not_before_time=not_before, not_after_time=not_after,
                    busy_cache=slot_cache,
                )
                timing["tools"].append({"name": "find_next_slot", "ms": round((time.perf_counter() - _t) * 1000)})
            except Exception as exc:
                return f"Couldn't search for slots: {exc}"

        if result.get("available"):
            start_iso = result["start_iso"]
            end_iso = result["end_iso"]
            result_date = result.get("date", date or today)
            date_label = _fmt_date(result_date)
            display = f"{date_label} at {_fmt_time(datetime.fromisoformat(start_iso))}\u2013{_fmt_time(datetime.fromisoformat(end_iso))}"
            slot_cache["_pending"] = {
                "title": title, "start_iso": start_iso,
                "end_iso": end_iso, "timezone": tz_name, "display": display,
            }
            return f"Found a {duration} min slot: {display}. Shall I book it as **{title}**?"
        else:
            if date_range:
                return (f"No {duration}-min slot found between "
                        f"{_fmt_date(date_range['start'])} and {_fmt_date(date_range['end'])}.")
            return f"No {duration}-min slot available on {_fmt_date(date)}."

    return None  # unknown \u2014 caller uses NL fallback


# â”€â”€ Step 3: NL fallback for unknown intent (~80 tokens) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fallback_reply(user_message: str, model: str, timing: dict) -> str:
    _t = time.perf_counter()
    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "You are a scheduling assistant. You can ONLY check calendar availability, "
                    "find free slots, and book meetings. "
                    "You CANNOT send reminders, emails, notifications, or do anything else. "
                    "If asked to do something outside scheduling, politely say you can only help with scheduling. "
                    "Reply in 1-2 sentences."
                )},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=80,
        )
        timing["groq_calls"].append({"iter": "fallback", "ms": round((time.perf_counter() - _t) * 1000)})
        return resp.choices[0].message.content or "I didn't understand that. Could you rephrase?"
    except Exception:
        return "I didn't understand that. Could you say that differently?"


# â”€â”€ Main entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_agent(
    user_message: str,
    history: list[dict],
    credentials: Credentials,
    user_timezone: str = "UTC",
    slot_cache: dict | None = None,
) -> tuple[str, list[dict], dict]:
    """
    Run one conversational turn of the scheduling agent.
    Returns: (reply, updated_history, timing)
    """
    t_agent = time.perf_counter()
    timing: dict = {"groq_calls": [], "tools": [], "total_ms": 0}
    if slot_cache is None:
        slot_cache = {}
    today = _today_str(user_timezone)
    model = MODEL_PRIMARY

    # Step 0: zero-token fast path for obvious confirm/cancel
    fast = _fast_intent(user_message)
    if fast:
        intent: dict = {"action": fast}
        log.info("[agent] fast intent: %s", fast)
    else:
        # Step 1: LLM intent extraction (~150 tokens)
        preprocessed = _preprocess_message(user_message, user_timezone)
        prev_context = slot_cache.get("_last_context", "")
        _t = time.perf_counter()
        try:
            intent = extract_intent(preprocessed, user_timezone, prev_context, model=model)
        except RateLimitError:
            log.warning("[agent] rate limit on intent extraction, retrying with %s", MODEL_FALLBACK)
            try:
                intent = extract_intent(preprocessed, user_timezone, prev_context, model=MODEL_FALLBACK)
                model = MODEL_FALLBACK
            except (RateLimitError, Exception):
                timing["total_ms"] = round((time.perf_counter() - t_agent) * 1000)
                return (
                    "I'm being rate limited right now. Please try again in a moment.",
                    history, timing,
                )
        except Exception as exc:
            log.error("[agent] intent extraction failed: %s", exc)
            intent = {"action": "unknown"}
        groq_ms = round((time.perf_counter() - _t) * 1000)
        timing["groq_calls"].append({"iter": 0, "ms": groq_ms})
        log.info("[agent] intent=%s (%.0fms)", json.dumps(intent)[:150], groq_ms)

    # Persist compact context for next turn's intent call (multi-turn carry-over)
    slot_cache["_last_context"] = _intent_to_context(intent)

    # Step 2: Python deterministic routing (0 LLM tokens)
    reply = _route(intent, credentials, slot_cache, user_timezone, today, timing)

    # Step 3: NL fallback only for unknown intent (~80 tokens)
    if reply is None:
        reply = _fallback_reply(user_message, model, timing)

    timing["total_ms"] = round((time.perf_counter() - t_agent) * 1000)
    log.debug("[timing] run_agent total=%dms", timing["total_ms"])
    log.info("[agent] reply: %s", reply[:120])

    updated_history = [
        *history,
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return reply, updated_history, timing
