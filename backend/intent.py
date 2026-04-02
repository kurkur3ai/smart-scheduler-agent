"""
intent.py — Step 1 of the 2-step scheduling pipeline.

LLM is given only:
  - System: ~10 tokens ("Extract scheduling intent as JSON.")
  - Dates: ~20 tokens (today, tomorrow, next-week range if relevant)
  - Prev: ~15 tokens (key fields from last turn, for context carry-over)
  - Schema: ~70 tokens (compact field list)
  - User message: ~30-50 tokens

Total input: ~150 tokens.  Output: ~100 tokens of JSON.
Compare to old tool-calling: ~4000 tokens/turn.
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from groq import Groq, RateLimitError

log = logging.getLogger("intent")

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"), timeout=15.0)
MODEL = "llama-3.1-8b-instant"
MODEL_FALLBACK = "llama-3.3-70b-versatile"

_SYSTEM = (
    "Extract scheduling intent as JSON. Output only valid JSON. "
    "IMPORTANT: 'before X' means not_after=X (deadline), NOT time=X. "
    "'after X' means not_before=X. time= is ONLY for exact start times."
)

# Compact schema hint
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
    # Examples: "before 6 PM" → not_after=18:00, time=null'
    # "at 3 PM" → time=15:00, no not_after'
    # "find me a slot tomorrow morning" → find_slot, not_before=08:00, time=null'
)


def _build_date_context(tz_name: str, message: str) -> str:
    """
    Build a small date-context string (~20-30 tokens) to anchor date references.
    Avoids sending a full 7-day map — only adds what the message actually needs.
    """
    tz = ZoneInfo(tz_name)
    today = datetime.now(tz).date()
    tomorrow = today + timedelta(days=1)
    ctx = f"today={today.isoformat()},tomorrow={tomorrow.isoformat()}"

    msg = message.lower()

    # Resolve named weekdays so LLM never has to guess
    _DAY_NAMES = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    for i, name in enumerate(_DAY_NAMES):
        if name in msg:
            # "next <day>" → at least 7 days out
            if f"next {name}" in msg:
                delta = (i - today.weekday()) % 7 or 7
            else:
                # nearest upcoming occurrence (today counts if same day)
                delta = (i - today.weekday()) % 7
            ctx += f",{name}={( today + timedelta(days=delta)).isoformat()}"

    if "next week" in msg:
        days_to_monday = (7 - today.weekday()) % 7 or 7
        nxt_mon = today + timedelta(days=days_to_monday)
        nxt_fri = nxt_mon + timedelta(days=4)
        ctx += f",next_week={nxt_mon.isoformat()} to {nxt_fri.isoformat()}"
    elif "this week" in msg:
        this_fri = today + timedelta(days=(4 - today.weekday()) % 7)
        ctx += f",this_week_end={this_fri.isoformat()}"

    return ctx


def _parse_raw(raw: str) -> dict:
    """Strip fences/comments and extract the first JSON object from LLM output."""
    raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    # Remove JS-style // comments that the model sometimes adds
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
    """
    Call LLM with a minimal prompt (~150 tokens input).

    prev_context: compact string from previous turn, e.g.
        "title=Call with John,date=2026-04-03,duration_minutes=60"
        Keeps multi-turn flow coherent without sending full history.
    """
    date_ctx = _build_date_context(tz_name, user_message)

    parts = [f"Dates:{date_ctx}"]
    if prev_context:
        parts.append(f"Prev:{prev_context}")
    parts.append(f"Schema:{_SCHEMA}")
    parts.append(f"Message:{user_message}")
    user_content = "\n".join(parts)

    raw = "{}"
    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content or "{}"
        return _parse_raw(raw)

    except json.JSONDecodeError as exc:
        log.warning("Intent JSON parse failed (%s): %r", exc, raw[:200])
        return {"action": "unknown"}
    except RateLimitError:
        raise
    except Exception as exc:
        log.error("Intent extraction error: %s", exc)
        return {"action": "unknown"}
