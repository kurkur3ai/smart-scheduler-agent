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
    mins = dt.strftime("%M")
    ampm = dt.strftime("%p")
    if mins == "00":
        return f"{hour} {ampm}"
    return f"{hour}:{mins} {ampm}"


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


def _apply_context_carryover(intent: dict, prev_context: str, user_message: str = "") -> dict:
    """
    Null-coalescing context: if the LLM returned null for a field, use the
    value from the previous turn.  If the LLM returned a non-null value the
    user explicitly stated something new, so LLM wins.

    This is simpler and more robust than word-detection heuristics.
    Only fires for slot-finding actions.
    """
    if not prev_context:
        return intent
    action = intent.get("action", "unknown")
    if action not in ("find_slot", "check_availability", "book_explicit"):
        return intent

    prev: dict = {}
    for part in prev_context.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            prev[k.strip()] = v.strip()

    # DATE — LLM null → use prev
    if not intent.get("date") and not intent.get("date_range"):
        if "date" in prev:
            intent["date"] = prev["date"]
            log.debug("[carryover] date ← prev: %s", prev["date"])
        elif "range" in prev:
            halves = prev["range"].split(" to ", 1)
            if len(halves) == 2:
                intent["date_range"] = {"start": halves[0], "end": halves[1]}
                log.debug("[carryover] date_range ← prev: %s", prev["range"])

    # DURATION — LLM null → use prev
    if not intent.get("duration_minutes") and "duration_minutes" in prev:
        try:
            intent["duration_minutes"] = int(prev["duration_minutes"])
            log.debug("[carryover] duration ← prev: %s min", prev["duration_minutes"])
        except ValueError:
            pass

    # TIME — LLM null → use prev (user established a specific start time earlier)
    # Clear any conflicting not_before/not_after — an exact time supersedes them.
    if not intent.get("time") and "time" in prev:
        intent["time"] = prev["time"]
        log.debug("[carryover] time ← prev: %s", prev["time"])

    # TITLE — LLM null → use prev
    if not intent.get("title") and "title" in prev:
        intent["title"] = prev["title"]

    # CONSTRAINTS — fill only the null slots from prev.
    # Skip not_before/not_after if an exact time was carried (they would conflict).
    constraints = intent.get("constraints") or {}
    if not intent.get("time"):
        if not constraints.get("not_before") and "not_before" in prev:
            constraints["not_before"] = prev["not_before"]
            log.debug("[carryover] not_before ← prev: %s", prev["not_before"])
        if not constraints.get("not_after") and "not_after" in prev:
            constraints["not_after"] = prev["not_after"]
            log.debug("[carryover] not_after ← prev: %s", prev["not_after"])
    intent["constraints"] = constraints

    return intent


#Step 0: zero-token fast intent detection 

_CONFIRM_WORDS = {
    "yes", "yeah", "yep", "sure", "ok", "okay", "go ahead",
    "confirm", "book it", "do it", "yup", "absolutely", "schedule it",
    "ye", "ya", "yea", "aye", "sounds good", "let's do it", "go for it",
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
        constraints = intent.get("constraints") or {}
        not_before = constraints.get("not_before")
        not_after  = constraints.get("not_after")

        if not specific_time:
            # General availability query — list free windows
            _t = time.perf_counter()
            try:
                windows = get_availability(creds, date, tz_name, slot_cache)
                timing["tools"].append({"name": "get_availability", "ms": round((time.perf_counter() - _t) * 1000)})
            except Exception as exc:
                return f"Couldn't check calendar: {exc}"
            if not windows:
                return f"You're fully booked on {date_label}."
            # Filter by time-of-day constraints (e.g. "afternoon" → not_before=12:00)
            tz = ZoneInfo(tz_name)
            def _parse_hhmm(hhmm: str) -> datetime | None:
                try:
                    h, m = hhmm.split(":")
                    d = datetime.strptime(date, "%Y-%m-%d").replace(
                        hour=int(h), minute=int(m), tzinfo=tz)
                    return d
                except Exception:
                    return None
            nb_dt = _parse_hhmm(not_before) if not_before else None
            na_dt = _parse_hhmm(not_after)  if not_after  else None
            if nb_dt or na_dt:
                filtered = []
                for w in windows:
                    w_from = datetime.fromisoformat(w["from"])
                    w_to   = datetime.fromisoformat(w["to"])
                    if nb_dt and w_to   <= nb_dt: continue
                    if na_dt and w_from >= na_dt: continue
                    # Clip window to the constraint range
                    clipped_from = max(w_from, nb_dt) if nb_dt else w_from
                    clipped_to   = min(w_to,   na_dt) if na_dt else w_to
                    if clipped_to > clipped_from:
                        filtered.append({"from": clipped_from.isoformat(),
                                         "to":   clipped_to.isoformat()})
                windows = filtered or windows  # fall back to all windows if filter removes everything
            fitting = [
                w for w in windows
                if (datetime.fromisoformat(w["to"]) - datetime.fromisoformat(w["from"])).seconds >= duration * 60
            ]
            # Format: avoid showing midnight as the endpoint — cap display at 10 PM
            def _display_window(w: dict) -> str:
                s = datetime.fromisoformat(w["from"])
                e = datetime.fromisoformat(w["to"])
                # If window runs to end-of-day midnight, show a readable end-of-workday cap
                e_naive_hour = e.hour
                e_display = "10 PM" if e_naive_hour == 0 and e.date() > s.date() else _fmt_time(e)
                return f"{_fmt_time(s)}–{e_display}"
            window_strs = ", ".join(_display_window(w) for w in (fitting or windows)[:4])
            return f"On {date_label} you're free: {window_strs}."

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
            if action == "check_availability":
                return f"{_fmt_time(s_dt)}\u2013{_fmt_time(e_dt)} is free on {date_label}. Want me to book it as **{title}**?"
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
            anchor_resolved = False
            try:
                anc = search_event(creds, anchor, anchor_search_start, anchor_search_end, tz_name)
                timing["tools"].append({"name": "search_event(anchor)", "ms": round((time.perf_counter() - _t) * 1000)})
                anc_events = anc.get("events", [])
                if anc_events:
                    ae = anc_events[0]
                    ae_date = ae["date"]
                    anchor_resolved = True
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
                else:
                    # Event not found in calendar — can't apply constraint
                    return f"I couldn't find '{anchor}' on your calendar. Could you check the event name or date?"
            except Exception as exc:
                log.warning("[router] anchor search failed: %s", exc)
                # Network/API error — can't safely proceed without the time constraint
                return f"I couldn't look up '{anchor}' right now (calendar error). Try again in a moment."

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
                    "You are a voice scheduling assistant. "
                    "You can ONLY check availability and book meetings. "
                    "Reply in one short sentence, 10 words max."
                )},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=40,
        )
        timing["groq_calls"].append({"iter": "fallback", "ms": round((time.perf_counter() - _t) * 1000)})
        return resp.choices[0].message.content or "I didn't understand that. Could you rephrase?"
    except Exception:
        return "I didn't understand that. Could you say that differently?"


# â”€â”€ Main entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_agent(
    user_message: str,
    credentials: Credentials,
    user_timezone: str = "UTC",
    slot_cache: dict | None = None,
) -> tuple[str, dict]:
    """
    Run one conversational turn of the scheduling agent.
    Returns: (reply, timing)
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
        prev_context = slot_cache.get("_last_context", "")
        _t = time.perf_counter()
        try:
            intent = extract_intent(user_message, user_timezone, prev_context, model=model)
        except RateLimitError:
            log.warning("[agent] rate limit on intent extraction, retrying with %s", MODEL_FALLBACK)
            try:
                intent = extract_intent(user_message, user_timezone, prev_context, model=MODEL_FALLBACK)
                model = MODEL_FALLBACK
            except (RateLimitError, Exception):
                timing["total_ms"] = round((time.perf_counter() - t_agent) * 1000)
                return (
                    "I'm being rate limited right now. Please try again in a moment.",
                    timing,
                )
        except Exception as exc:
            log.error("[agent] intent extraction failed: %s", exc)
            intent = {"action": "unknown"}
        groq_ms = round((time.perf_counter() - _t) * 1000)
        timing["groq_calls"].append({"iter": 0, "ms": groq_ms})
        log.info("[agent] intent=%s (%.0fms)", json.dumps(intent)[:150], groq_ms)

    # For refinement turns, fill null fields from prev turn's context
    if not fast:
        intent = _apply_context_carryover(intent, prev_context)

    # Persist compact context for next turn's intent call (multi-turn carry-over).
    # Skip on confirm/cancel — those are meta-actions that carry no new scheduling
    # context, so we preserve whatever date/duration was already established.
    # This prevents VAD-split speech ("No." then "check 1 PM") from losing the date.
    if intent.get("action") not in ("confirm", "cancel"):
        slot_cache["_last_context"] = _intent_to_context(intent)

    # Step 2: Python deterministic routing (0 LLM tokens)
    reply = _route(intent, credentials, slot_cache, user_timezone, today, timing)

    # Step 3: NL fallback only for unknown intent (~80 tokens)
    if reply is None:
        pending = slot_cache.get("_pending")
        if pending:
            reply = f"Did you want to confirm booking **{pending['title']}** at {pending['display']}?"
        else:
            reply = _fallback_reply(user_message, model, timing)

    timing["total_ms"] = round((time.perf_counter() - t_agent) * 1000)
    log.debug("[timing] run_agent total=%dms", timing["total_ms"])
    log.info("[agent] reply: %s", reply[:120])
    return reply, timing
