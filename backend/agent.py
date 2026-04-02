# agent.py — LLM conversation + tool calling logic

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from groq import BadRequestError, Groq, RateLimitError
from google.oauth2.credentials import Credentials

from calendar_tool import (
    get_availability, check_slot, book_slot, find_next_slot, compute_free_windows,
    get_events_for_day, search_event, find_slot_in_range,
)
from prompts import SYSTEM_PROMPT

log = logging.getLogger("agent")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"), timeout=20.0)

MODEL_PRIMARY  = "llama-3.1-8b-instant"
MODEL_FALLBACK = "llama-3.3-70b-versatile"
MODEL = MODEL_PRIMARY
MAX_TOOL_ITERATIONS = 5
MAX_HISTORY_TURNS = 6   # keep last N user+assistant pairs to limit token usage

# Tool schemas
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_free_windows",
            "description": (
                "List all free time blocks for a full calendar day. "
                "USE WHEN: user asks about availability without a specific time, "
                "or when you need options to offer. "
                "Do NOT call if this date is already in CALENDAR CACHE. "
                "Call at most once per turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Kolkata."},
                },
                "required": ["date", "timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_next_slot",
            "description": (
                "Find the first available time slot of a given duration on a specific day. "
                "USE WHEN: user wants a slot found for them and provides a duration but no exact time. "
                "Handles morning/afternoon/evening via not_before_time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "duration_minutes": {"type": "integer", "description": "Meeting duration in minutes."},
                    "timezone": {"type": "string", "description": "IANA timezone."},
                    "not_before_time": {
                        "type": "string",
                        "description": (
                            "Earliest acceptable start time as HH:MM in user's timezone. Optional. "
                            "E.g. '09:00' for morning, '12:00' for afternoon, '17:00' for evening."
                        ),
                    },
                    "not_after_time": {
                        "type": "string",
                        "description": (
                            "Latest acceptable end time as HH:MM in user's timezone. Optional. "
                            "Defaults to 24:00. E.g. '18:00' means the meeting must end by 6 PM."
                        ),
                    },
                },
                "required": ["date", "duration_minutes", "timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_slot_available",
            "description": (
                "Check whether a specific time range is free on the calendar. Returns available=true/false. "
                "USE WHEN: you have an exact start and end time and need to verify before showing the confirmation. "
                "Reuses cached day data — no extra Google API call if the same date was already fetched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_iso": {
                        "type": "string",
                        "description": "ISO 8601 start datetime with timezone offset, e.g. 2026-03-30T15:00:00+05:30",
                    },
                    "end_iso": {
                        "type": "string",
                        "description": "ISO 8601 end datetime with timezone offset.",
                    },
                    "timezone": {"type": "string", "description": "IANA timezone."},
                },
                "required": ["start_iso", "end_iso", "timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_event",
            "description": (
                "Create a calendar event. "
                "ONLY call this after the user has EXPLICITLY confirmed (said yes/go ahead/confirm). "
                "Never call speculatively or before user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title / summary."},
                    "start_iso": {"type": "string", "description": "ISO 8601 start with timezone offset."},
                    "end_iso": {"type": "string", "description": "ISO 8601 end with timezone offset."},
                    "timezone": {"type": "string", "description": "IANA timezone."},
                    "description": {"type": "string", "description": "Optional event description."},
                },
                "required": ["title", "start_iso", "end_iso", "timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events",
            "description": (
                "Get the list of calendar events with titles and times for a specific day. "
                "USE WHEN: user asks what's on their calendar, wants to know event names, "
                "or you need an event name as a time reference. "
                "Reuses cached data — no extra API call if day was already fetched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "timezone": {"type": "string", "description": "IANA timezone."},
                },
                "required": ["date", "timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_slot_in_range",
            "description": (
                "Find the first available slot across a date range, fetching days in parallel. "
                "USE WHEN: user says 'sometime next week', 'any day this week', "
                "or gives a range with constraints like 'not on Wednesday'. "
                "More efficient than calling find_next_slot per day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start of range in YYYY-MM-DD."},
                    "end_date": {"type": "string", "description": "End of range in YYYY-MM-DD."},
                    "duration_minutes": {"type": "integer", "description": "Meeting duration in minutes."},
                    "timezone": {"type": "string", "description": "IANA timezone."},
                    "not_before_time": {
                        "type": "string",
                        "description": "Earliest start time each day as HH:MM. Defaults to 08:00.",
                    },
                    "not_after_time": {
                        "type": "string",
                        "description": "Latest end time each day as HH:MM. Defaults to 22:00.",
                    },
                    "exclude_days": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lowercase day names to skip, e.g. ['wednesday', 'saturday'].",
                    },
                },
                "required": ["start_date", "end_date", "duration_minutes", "timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_event",
            "description": (
                "Search for calendar events by keyword/title across a date range. "
                "USE WHEN: user references an event by name, e.g. 'after my Project Alpha kickoff' "
                "or 'before my flight on Friday'. Returns matching event titles and times."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword to search for in event titles."},
                    "start_date": {"type": "string", "description": "Start of search range in YYYY-MM-DD."},
                    "end_date": {"type": "string", "description": "End of search range in YYYY-MM-DD."},
                    "timezone": {"type": "string", "description": "IANA timezone."},
                },
                "required": ["query", "start_date", "end_date", "timezone"],
            },
        },
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(dt) -> str:
    """Format a datetime to '3:00 PM' style (cross-platform, no leading zero)."""
    hour = int(dt.strftime("%I"))  # strftime %I gives 01-12; cast strips the zero
    return f"{hour}:{dt.strftime('%M %p')}"


def _summarize_tool_result(name: str, result: dict) -> str:
    """Convert a tool result into a compact human-readable string for the model context."""
    if "error" in result:
        return f"Error: {result['error']}"

    if name == "get_free_windows":
        windows = result.get("free_windows", [])
        if not windows:
            return f"No free windows on {result.get('date', '?')}."
        try:
            parts = []
            for w in windows:
                s = datetime.fromisoformat(w["from"])
                e = datetime.fromisoformat(w["to"])
                parts.append(f"{_fmt_time(s)}\u2013{_fmt_time(e)}")
            return f"Free on {result['date']}: {', '.join(parts)}"
        except Exception:
            return json.dumps(result)

    if name == "find_next_slot":
        if result.get("available"):
            try:
                s = datetime.fromisoformat(result["start_iso"])
                e = datetime.fromisoformat(result["end_iso"])
                return f"Next available {result.get('duration_minutes', '?')}-min slot on {result['date']}: {_fmt_time(s)}\u2013{_fmt_time(e)}"
            except Exception:
                return json.dumps(result)
        return f"No available {result.get('duration_minutes', '?')}-min slot found on {result.get('date', '?')}."

    if name == "check_slot_available":
        if result.get("available"):
            return "Slot is free."
        conflict = result.get("conflict", {})
        try:
            cs = datetime.fromisoformat(conflict["start"])
            ce = datetime.fromisoformat(conflict["end"])
            return f"Slot is taken (conflict: {_fmt_time(cs)}\u2013{_fmt_time(ce)})."
        except Exception:
            return "Slot is taken."

    if name == "book_event":
        if result.get("status") == "booked":
            return (
                f"Event booked: '{result.get('summary')}' "
                f"at {result.get('start')}\u2013{result.get('end')}. "
                f"Link: {result.get('html_link')}"
            )

    if name == "get_events":
        events = result.get("events", [])
        if not events:
            return f"No events on {result.get('date', '?')}."
        try:
            parts = []
            for e in events:
                s = datetime.fromisoformat(e["start"])
                et = datetime.fromisoformat(e["end"])
                parts.append(f"{_fmt_time(s)}\u2013{_fmt_time(et)}: {e['title']}")
            return f"Events on {result['date']}: " + "; ".join(parts)
        except Exception:
            return json.dumps(result)

    if name == "find_slot_in_range":
        if result.get("available"):
            try:
                s = datetime.fromisoformat(result["start_iso"])
                e = datetime.fromisoformat(result["end_iso"])
                return (
                    f"First available {result.get('duration_minutes', '?')}-min slot: "
                    f"{result['date']} at {_fmt_time(s)}\u2013{_fmt_time(e)}"
                )
            except Exception:
                return json.dumps(result)
        return (
            f"No available {result.get('duration_minutes', '?')}-min slot found "
            f"between {result.get('start_date', '?')} and {result.get('end_date', '?')}."
        )

    if name == "search_event":
        events = result.get("events", [])
        if not events:
            return f"No events found matching '{result.get('query', '?')}'"
        try:
            parts = []
            for e in events:
                s_str = e.get("start", "")
                time_str = _fmt_time(datetime.fromisoformat(s_str)) if "T" in s_str else s_str[:10]
                parts.append(f"{e['date']}: '{e['title']}' at {time_str}")
            return f"Found {len(events)} event(s): " + "; ".join(parts)
        except Exception:
            return json.dumps(result)

    return json.dumps(result)


def _build_cache_section(slot_cache: dict, user_timezone: str) -> str:
    """Build the CALENDAR CACHE block to append to the system prompt."""
    if not slot_cache:
        return ""
    now_t = time.time()
    lines = []
    for bkey, entry in slot_cache.items():
        if now_t >= entry.get("expires", 0) or "busy" not in entry:
            continue
        date_str, tz_str = bkey.split(":", 1)
        tz = ZoneInfo(tz_str)
        day = datetime.strptime(date_str, "%Y-%m-%d")
        day_s = day.replace(hour=0, minute=0, second=0, tzinfo=tz)
        day_e = (day + timedelta(days=1)).replace(hour=0, minute=0, second=0, tzinfo=tz)
        now_local = datetime.now(tz)
        if day.date() == now_local.date():
            mins = now_local.minute
            ru = timedelta(minutes=(-mins % 15))
            nr = now_local.replace(second=0, microsecond=0) + ru
            if nr > day_s:
                day_s = nr
        windows = compute_free_windows(entry["busy"], day_s, day_e, tz, 15)
        if windows:
            w_strs = [
                f"{_fmt_time(datetime.fromisoformat(w['from']))}\u2013{_fmt_time(datetime.fromisoformat(w['to']))}"
                for w in windows
            ]
            lines.append(f"\u2022 {date_str}: free {', '.join(w_strs)}")
        else:
            lines.append(f"\u2022 {date_str}: fully booked")
    if not lines:
        return ""
    return (
        "\n\nCALENDAR CACHE \u2014 already fetched, do NOT call get_free_windows, get_events, or check_slot_available for these dates:\n"
        + "\n".join(lines)
    )


# ── Tool dispatcher ───────────────────────────────────────────────────────────

def _execute_tool(name: str, args: dict, creds: Credentials, slot_cache: dict | None = None) -> dict:
    """Dispatch a tool call to the corresponding calendar function."""
    log.info("[tool] %s  args=%s", name, json.dumps(args))

    if name == "get_free_windows":
        try:
            windows = get_availability(
                creds=creds,
                date=args["date"],
                tz_name=args["timezone"],
                busy_cache=slot_cache,
            )
            return {"date": args["date"], "free_windows": windows, "count": len(windows)}
        except Exception as exc:
            log.error("[tool] get_free_windows error: %s", exc)
            return {"error": str(exc)}

    if name == "find_next_slot":
        try:
            return find_next_slot(
                creds=creds,
                date=args["date"],
                duration_minutes=int(args["duration_minutes"]),
                tz_name=args["timezone"],
                not_before_time=args.get("not_before_time"),
                not_after_time=args.get("not_after_time"),
                busy_cache=slot_cache,
            )
        except Exception as exc:
            log.error("[tool] find_next_slot error: %s", exc)
            return {"error": str(exc)}

    if name == "check_slot_available":
        try:
            tz = ZoneInfo(args["timezone"])
            slot_start = datetime.fromisoformat(args["start_iso"]).astimezone(tz)
            date_str = slot_start.strftime("%Y-%m-%d")
            return check_slot(
                creds=creds,
                date=date_str,
                start_iso=args["start_iso"],
                end_iso=args["end_iso"],
                tz_name=args["timezone"],
                busy_cache=slot_cache,
            )
        except Exception as exc:
            log.error("[tool] check_slot_available error: %s", exc)
            return {"error": str(exc)}

    if name == "book_event":
        try:
            event = book_slot(
                creds=creds,
                start=args["start_iso"],
                end=args["end_iso"],
                summary=args["title"],
                description=args.get("description", ""),
            )
            return {
                "status": "booked",
                "event_id": event.get("id"),
                "html_link": event.get("htmlLink"),
                "summary": event.get("summary"),
                "start": event.get("start", {}).get("dateTime"),
                "end": event.get("end", {}).get("dateTime"),
            }
        except Exception as exc:
            log.error("[tool] book_event error: %s", exc)
            return {"error": str(exc)}

    if name == "get_events":
        try:
            events = get_events_for_day(
                creds=creds,
                date=args["date"],
                tz_name=args["timezone"],
                busy_cache=slot_cache,
            )
            return {"date": args["date"], "events": events, "count": len(events)}
        except Exception as exc:
            log.error("[tool] get_events error: %s", exc)
            return {"error": str(exc)}

    if name == "find_slot_in_range":
        try:
            return find_slot_in_range(
                creds=creds,
                start_date=args["start_date"],
                end_date=args["end_date"],
                duration_minutes=int(args["duration_minutes"]),
                tz_name=args["timezone"],
                not_before_time=args.get("not_before_time"),
                not_after_time=args.get("not_after_time"),
                exclude_days=args.get("exclude_days"),
                busy_cache=slot_cache,
            )
        except Exception as exc:
            log.error("[tool] find_slot_in_range error: %s", exc)
            return {"error": str(exc)}

    if name == "search_event":
        try:
            return search_event(
                creds=creds,
                query=args["query"],
                start_date=args["start_date"],
                end_date=args["end_date"],
                tz_name=args["timezone"],
            )
        except Exception as exc:
            log.error("[tool] search_event error: %s", exc)
            return {"error": str(exc)}

    return {"error": f"Unknown tool: {name}"}


def run_agent(
    user_message: str,
    history: list[dict],
    credentials: Credentials,
    user_timezone: str = "UTC",
    slot_cache: dict | None = None,
) -> tuple[str, list[dict], dict]:
    """
    Run one conversational turn of the scheduling agent.

    Returns:
        (assistant_reply: str, updated_history: list[dict], timing: dict)
        timing keys: total_ms, groq_calls [{iter, ms}], tools [{name, ms}]
    """
    t_agent = time.perf_counter()
    _timing: dict = {"groq_calls": [], "tools": []}

    # Build concrete day→date map for the next 7 days so the model never has to compute it
    tz = ZoneInfo(user_timezone)
    _now_local = datetime.now(tz)
    _day_map_lines = []
    for i in range(7):
        d = (_now_local + timedelta(days=i)).date()
        label = "today" if i == 0 else ("tomorrow" if i == 1 else "")
        suffix = f" ({label})" if label else ""
        _day_map_lines.append(f"  {d.strftime('%A')} = {d.strftime('%Y-%m-%d')}{suffix}")
    _day_map = "UPCOMING DATES (use these exact dates — do not compute):\n" + "\n".join(_day_map_lines)

    today = _now_local.strftime("%A, %B %d, %Y")

    # Trim history to last MAX_HISTORY_TURNS user+assistant pairs to control token usage
    trimmed_history = history[-(MAX_HISTORY_TURNS * 2):]

    def _system_msg() -> dict:
        content = SYSTEM_PROMPT.format(today=today, timezone=user_timezone)
        content += "\n\n" + _day_map
        content += _build_cache_section(slot_cache or {}, user_timezone)
        return {"role": "system", "content": content}

    log.info("[agent] user: %s", user_message[:120])

    messages: list = [
        _system_msg(),
        *trimmed_history,
        {"role": "user", "content": user_message},
    ]
    _force_text_next = False  # set after find_next_slot/find_slot_in_range to block further tool calls

    for iteration in range(MAX_TOOL_ITERATIONS):
        force_text = _force_text_next or iteration == MAX_TOOL_ITERATIONS - 1

        try:
            _t_groq = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="none" if force_text else "auto",
                temperature=0.2,
                max_tokens=512,
            )
            _groq_ms = round((time.perf_counter() - _t_groq) * 1000)
            _timing["groq_calls"].append({"iter": iteration, "ms": _groq_ms})
            log.debug("[timing] groq iter=%d  %dms", iteration, _groq_ms)
        except BadRequestError as exc:
            # Groq 400: model leaked tool-call syntax into content instead of using tool_calls.
            # Retry this same iteration with tool_choice="none" to get a clean text reply.
            if "tool call validation failed" in str(exc) and not force_text:
                log.warning("[agent] Groq tool-leak 400 on iter=%d, retrying text-only", iteration)
                try:
                    _t_groq = time.perf_counter()
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="none",
                        temperature=0.2,
                        max_tokens=512,
                    )
                    _groq_ms = round((time.perf_counter() - _t_groq) * 1000)
                    _timing["groq_calls"].append({"iter": iteration, "ms": _groq_ms})
                    log.debug("[timing] groq iter=%d (retry)  %dms", iteration, _groq_ms)
                except Exception as retry_exc:
                    raise RuntimeError(f"Groq API error: {retry_exc}") from retry_exc
            else:
                raise RuntimeError(f"Groq API error: {exc}") from exc
        except RateLimitError as exc:
            # Swap to fallback model and retry once rather than waiting for SDK backoff
            fallback = MODEL_FALLBACK if MODEL == MODEL_PRIMARY else MODEL_PRIMARY
            log.warning("[agent] Groq rate-limit on %s iter=%d, retrying with %s", MODEL, iteration, fallback)
            try:
                _t_groq = time.perf_counter()
                response = client.chat.completions.create(
                    model=fallback,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="none" if force_text else "auto",
                    temperature=0.2,
                    max_tokens=512,
                )
                _groq_ms = round((time.perf_counter() - _t_groq) * 1000)
                _timing["groq_calls"].append({"iter": iteration, "ms": _groq_ms})
                log.debug("[timing] groq iter=%d (fallback=%s)  %dms", iteration, fallback, _groq_ms)
            except Exception as retry_exc:
                raise RuntimeError(f"Groq API error (fallback): {retry_exc}") from retry_exc
        except Exception as exc:
            raise RuntimeError(f"Groq API error: {exc}") from exc

        choice = response.choices[0]
        msg = choice.message

        # No tool calls — model produced a final text response
        if not msg.tool_calls:
            reply = msg.content or ""
            # Strip leaked function-call syntax the model sometimes emits as plain text
            reply = re.sub(r'\(function=\w+>[^)]*\)', '', reply)
            reply = re.sub(r'<function_calls>.*?</function_calls>', '', reply, flags=re.DOTALL)
            reply = reply.strip()
            _timing["total_ms"] = round((time.perf_counter() - t_agent) * 1000)
            log.debug("[timing] run_agent total=%dms iters=%d", _timing["total_ms"], iteration + 1)
            log.info("[agent] reply: %s", reply[:120])
            return (
                reply,
                [*history, {"role": "user", "content": user_message}, {"role": "assistant", "content": reply}],
                _timing,
            )

        log.debug("[agent] tool calls: %s", [tc.function.name for tc in msg.tool_calls])

        # Append the assistant turn (content must be None when tool_calls present — Groq requirement)
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute tools; deduplicate identical calls; limit get_free_windows to 1 per iteration
        seen_calls: dict[str, dict] = {}
        free_windows_count = 0
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if tc.function.name == "get_free_windows":
                free_windows_count += 1
                if free_windows_count > 1:
                    log.debug("[tool] suppressed duplicate get_free_windows for %s", args.get("date"))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "suppressed: only 1 get_free_windows allowed per turn",
                    })
                    continue

            dedup_key = tc.function.name + json.dumps(args, sort_keys=True)
            if dedup_key in seen_calls:
                log.debug("[tool] dedup skip %s", tc.function.name)
                result = seen_calls[dedup_key]
            else:
                _t_tool = time.perf_counter()
                result = _execute_tool(tc.function.name, args, credentials, slot_cache)
                _tool_ms = round((time.perf_counter() - _t_tool) * 1000)
                _timing["tools"].append({"name": tc.function.name, "ms": _tool_ms})
                log.debug("[timing] tool=%s  %dms", tc.function.name, _tool_ms)
                seen_calls[dedup_key] = result

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _summarize_tool_result(tc.function.name, result),
            })

        # Short-circuit: if book_event succeeded, build reply directly — no LLM round-trip needed
        booked_results = [
            seen_calls[k] for k in seen_calls
            if k.startswith("book_event") and seen_calls[k].get("status") == "booked"
        ]
        if booked_results:
            r = booked_results[0]
            try:
                s = datetime.fromisoformat(r["start"])
                e = datetime.fromisoformat(r["end"])
                reply = (
                    f"Done! {r['summary']} is on your calendar for "
                    f"{s.strftime('%A, %B')} {s.day} at {_fmt_time(s)}\u2013{_fmt_time(e)}."
                )
            except Exception:
                reply = f"Done! {r.get('summary', 'Event')} has been booked."
            _timing["total_ms"] = round((time.perf_counter() - t_agent) * 1000)
            log.debug("[timing] run_agent total=%dms iters=%d (book shortcut)", _timing["total_ms"], iteration + 1)
            log.info("[agent] reply: %s", reply)
            return (
                reply,
                [*history, {"role": "user", "content": user_message}, {"role": "assistant", "content": reply}],
                _timing,
            )

        # If find_next_slot or find_slot_in_range returned a free slot, force the next LLM call
        # to text-only — the model will write the confirmation naturally but can't call more tools.
        slot_found = any(
            (k.startswith("find_next_slot") or k.startswith("find_slot_in_range"))
            and seen_calls[k].get("available") is True
            for k in seen_calls
        )
        if slot_found:
            _force_text_next = True
            log.debug("[timing] slot found — forcing text-only on next iteration")
            # Inject a hint so the model skips STEP 2 collection and goes straight to confirmation
            messages.append({
                "role": "user",
                "content": (
                    "[SYSTEM HINT: A free slot was found by the tool above. "
                    "You have all required information. "
                    "Skip to STEP 4 — present the slot for confirmation. "
                    "Do NOT ask for more details.]"
                ),
            })

        # Refresh system message so subsequent iterations see the updated cache
        messages[0] = _system_msg()

    raise RuntimeError("Agent exceeded maximum tool iterations without producing a final reply.")
