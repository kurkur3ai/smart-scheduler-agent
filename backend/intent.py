"""
intent.py — Step 1 of the 2-step scheduling pipeline.

Normal turn (simple date like "tomorrow" or "Friday"):
  1 LLM call, ~150 tokens in / ~100 tokens out.

Turn with a relative date the LLM can't resolve from today/tomorrow alone:
  LLM calls get_date_map(start?, end?) → Python returns {YYYY-MM-DD: "Weekday D Mon"}
  LLM reads the map and picks the right date → fills JSON.
  2 LLM calls total.

Why get_date_map instead of resolve_date(anchor, offset):
  - LLM reads dates directly — no fixed anchor enum, no abstraction ceiling.
  - Handles ANY expression: "second Tuesday of next month", "last weekday before
    the 15th", "three Fridays from now", "the Monday after next", etc.
  - Adding new phrasings needs zero code changes.
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta, date as _date
from zoneinfo import ZoneInfo

from groq import Groq, RateLimitError

log = logging.getLogger("intent")

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"), timeout=15.0)
MODEL = "llama-3.1-8b-instant"
MODEL_FALLBACK = "llama-3.3-70b-versatile"

_SYSTEM = (
    "Extract scheduling intent as JSON. Output only valid JSON. "
    "IMPORTANT: 'before X' means not_after=X (deadline), NOT time=X. "
    "'after X' means not_before=X. time= is ONLY for exact start times. "
    "CRITICAL: date=null if the user did NOT mention a date, day, or time period — "
    "never default to today unless the user explicitly said 'today'. "
    "Use the get_date_map tool whenever the user says something like "
    "'last weekday of next week', 'end of next month', 'second Tuesday of May', "
    "or any other date you cannot compute from today/tomorrow alone."
)

_SCHEMA = (
    '{"action":"check_availability|find_slot|book_explicit|list_events'
    '|search_event|confirm|cancel|unknown",'
    '"title":str|null,"date":"YYYY-MM-DD"|null,'
    '"date_range":{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}|null,'
    '"time":"HH:MM"|null,'
    '"duration_minutes":int|null,'
    '"constraints":{"not_before":"HH:MM"|null,"not_after":"HH:MM"|null,"exclude_days":[]},'
    '"anchor_event":str|null,"anchor_relation":"before"|"after"|null,'
    '"anchor_offset_days":int|null,'
    '"buffer_minutes":int|null}'
)

# ── get_date_map tool ─────────────────────────────────────────────────────────

_GET_DATE_MAP_TOOL = {
    "type": "function",
    "function": {
        "name": "get_date_map",
        "description": (
            "Return a map of {YYYY-MM-DD: \'Weekday D Mon YYYY\'} for every date in the "
            "given range so you can identify the correct date for a relative expression. "
            "Call this when the user says something like 'last weekday of next week', "
            "'end of next month', 'second Tuesday of May', or any date you cannot "
            "determine from today/tomorrow alone. "
            "Omit start to default to today. Omit end to default to 60 days from start."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "string",
                    "description": (
                        "First date of the range, YYYY-MM-DD. "
                        "Defaults to today if omitted."
                    ),
                },
                "end": {
                    "type": "string",
                    "description": (
                        "Last date of the range (inclusive), YYYY-MM-DD. "
                        "Defaults to 60 days after start if omitted. "
                        "Keep the range as small as needed — the tool returns every day."
                    ),
                },
            },
            "required": [],
        },
    },
}

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _build_date_map(start_str: str | None, end_str: str | None, tz_name: str) -> str:
    """
    Build the date map returned to the LLM.

    Defaults:
      start → today in user's timezone
      end   → 60 days after start  (wide enough for 'end of next month',
               'third Friday of next quarter', etc.)

    Cap: never return more than 90 days to keep token count reasonable.
    Returns JSON: {"YYYY-MM-DD": "Weekday D Mon YYYY", ...}
    """
    tz = ZoneInfo(tz_name)
    today = datetime.now(tz).date()

    try:
        start = _date.fromisoformat(start_str) if start_str else today
    except ValueError:
        start = today

    try:
        end = _date.fromisoformat(end_str) if end_str else start + timedelta(days=60)
    except ValueError:
        end = start + timedelta(days=60)

    # Safety: start must not be in the past relative to today
    if start < today:
        start = today

    # If end is now before start (e.g. caller gave a past range), extend to +60d
    if end < start:
        end = start + timedelta(days=60)

    # Cap range at 90 days to avoid token explosion
    if (end - start).days > 90:
        end = start + timedelta(days=90)

    # Build map
    result: dict[str, str] = {}
    cur = start
    while cur <= end:
        label = f"{_DAY_NAMES[cur.weekday()]} {cur.day} {_MONTH_NAMES[cur.month - 1]} {cur.year}"
        result[cur.isoformat()] = label
        cur += timedelta(days=1)

    return json.dumps(result)


def _parse_raw(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    raw = re.sub(r"//[^\n]*", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    return json.loads(raw)


def extract_intent(
    user_message: str,
    tz_name: str = "UTC",
    prev_context: str = "",
    model: str = MODEL,
) -> dict:
    tz = ZoneInfo(tz_name)
    today_str    = datetime.now(tz).date().isoformat()
    tomorrow_str = (datetime.now(tz).date() + timedelta(days=1)).isoformat()

    parts = [f"Dates:today={today_str},tomorrow={tomorrow_str}"]
    if prev_context:
        parts.append(f"Prev:{prev_context}")
    parts.append(f"Schema:{_SCHEMA}")
    parts.append(f"Message:{user_message}")
    user_content = "\n".join(parts)

    messages: list = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    raw = "{}"
    try:
        for _ in range(3):  # at most 2 tool calls then final JSON answer
            resp = _client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[_GET_DATE_MAP_TOOL],
                tool_choice="auto",
                temperature=0,
                max_tokens=300,
            )
            assistant_msg = resp.choices[0].message

            if assistant_msg.tool_calls:
                # Keep assistant turn with tool_call IDs for correlation
                messages.append({
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_msg.tool_calls
                    ],
                })
                for tc in assistant_msg.tool_calls:
                    if tc.function.name == "get_date_map":
                        try:
                            args      = json.loads(tc.function.arguments)
                            start_arg = args.get("start")
                            end_arg   = args.get("end")
                            result    = _build_date_map(start_arg, end_arg, tz_name)
                            n_dates   = result.count(":")
                            log.debug(
                                "[intent] get_date_map(%s → %s) → %d dates",
                                start_arg or "today", end_arg or "+60d", n_dates,
                            )
                        except Exception as exc:
                            result = json.dumps({"error": str(exc)})
                            log.warning("[intent] get_date_map error: %s", exc)
                        messages.append({
                            "role":         "tool",
                            "tool_call_id": tc.id,
                            "content":      result,
                        })
                continue  # send map back, LLM picks date and returns JSON

            # No tool call → final answer, parse and return
            raw = assistant_msg.content or "{}"
            return _parse_raw(raw)

        log.warning("[intent] no final answer after tool calls")
        return {"action": "unknown"}

    except json.JSONDecodeError as exc:
        log.warning("Intent JSON parse failed (%s): %r", exc, raw[:200])
        return {"action": "unknown"}
    except RateLimitError:
        raise
    except Exception as exc:
        log.error("Intent extraction error: %s", exc)
        return {"action": "unknown"}
