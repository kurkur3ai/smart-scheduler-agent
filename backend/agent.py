# agent.py — LLM conversation + tool calling logic

import json
import os
from datetime import datetime, timezone

from groq import Groq
from google.oauth2.credentials import Credentials

from calendar_tool import get_free_slots, book_slot
from prompts import SYSTEM_PROMPT

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
MAX_TOOL_ITERATIONS = 6  # safety ceiling to prevent infinite loops

# Tool schemas sent to Groq
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_free_slots",
            "description": (
                "Query the user's Google Calendar and return available time slots on a "
                "specific date. ALWAYS call this before suggesting any meeting times. "
                "Never invent or guess availability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to check in YYYY-MM-DD format.",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Desired meeting duration in minutes.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": (
                            "IANA timezone name, e.g. America/New_York. "
                            "Use the user's timezone from the system prompt."
                        ),
                    },
                    "work_start_hour": {
                        "type": "integer",
                        "description": "First bookable hour (24h). Use 12 for afternoon, 8 for full day.",
                    },
                    "work_end_hour": {
                        "type": "integer",
                        "description": "Meetings must end by this hour (24h). Use 17 for afternoon.",
                    },
                },
                "required": ["date", "duration_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_slot",
            "description": (
                "Create a calendar event for the user. Only call this AFTER the user "
                "has explicitly confirmed they want to book a specific slot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "ISO 8601 start datetime with timezone offset, e.g. 2026-04-02T14:00:00-05:00",
                    },
                    "end": {
                        "type": "string",
                        "description": "ISO 8601 end datetime with timezone offset.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Event title.",
                    },
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of attendee email addresses. Optional.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional event description or notes.",
                    },
                },
                "required": ["start", "end", "summary"],
            },
        },
    },
]


def _execute_tool(name: str, args: dict, creds: Credentials) -> dict:
    """
    Dispatch a tool call to the corresponding calendar function.
    Returns a result dict that is JSON-serialisable.
    """
    if name == "get_free_slots":
        try:
            slots = get_free_slots(
                creds=creds,
                date=args["date"],
                duration_minutes=args["duration_minutes"],
                tz_name=args.get("timezone", "UTC"),
                work_start_hour=args.get("work_start_hour", 8),
                work_end_hour=args.get("work_end_hour", 18),
            )
            return {
                "date": args["date"],
                "duration_minutes": args["duration_minutes"],
                "available_slots": slots,
                "count": len(slots),
            }
        except Exception as exc:
            return {"error": str(exc)}

    if name == "book_slot":
        try:
            event = book_slot(
                creds=creds,
                start=args["start"],
                end=args["end"],
                summary=args["summary"],
                attendees=args.get("attendees"),
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
            return {"error": str(exc)}

    return {"error": f"Unknown tool: {name}"}


def run_agent(
    user_message: str,
    history: list[dict],
    credentials: Credentials,
    user_timezone: str = "UTC",
) -> tuple[str, list[dict]]:
    """
    Run one conversational turn of the scheduling agent.

    Args:
        user_message:  The user's latest text input.
        history:       Prior turns as [{"role": ..., "content": ...}, ...]
                       (only user + assistant text messages, no tool internals).
        credentials:   Google OAuth2 credentials for this user's session.
        user_timezone: IANA timezone string, e.g. "America/New_York".

    Returns:
        (assistant_reply: str, updated_history: list[dict])

    Raises:
        RuntimeError: If the Groq API call fails or the tool loop exceeds the
                      safety ceiling.
    """
    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")
    system_content = SYSTEM_PROMPT.format(today=today, timezone=user_timezone)

    # Build the full message list for this turn
    messages: list = [
        {"role": "system", "content": system_content},
        *history,
        {"role": "user", "content": user_message},
    ]

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as exc:
            raise RuntimeError(f"Groq API error: {exc}") from exc

        choice = response.choices[0]
        msg = choice.message

        # No tool calls — model produced a final text response
        if not msg.tool_calls:
            reply = msg.content or ""
            updated_history = [
                *history,
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": reply},
            ]
            return reply, updated_history

        # Append the assistant message (with tool_calls) to the working context
        assistant_dict: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_dict)

        # Execute every tool the model requested and feed results back
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = _execute_tool(tc.function.name, args, credentials)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    # Safety fallback — should not normally be reached
    raise RuntimeError(
        "Agent exceeded maximum tool iterations without producing a final reply."
    )
