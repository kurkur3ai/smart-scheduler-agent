# Smart Scheduler

A conversational AI scheduling agent that connects to Google Calendar and lets users check availability, find free slots, and book meetings through natural language — via text chat or voice.

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Request Flow](#architecture--request-flow)
- [Project Structure](#project-structure)
- [Features](#features)
- [How the Agent Pipeline Works](#how-the-agent-pipeline-works)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Local Setup](#local-setup)
- [Deployment (Render)](#deployment-render)
- [Tech Stack](#tech-stack)

---

## Overview

Smart Scheduler is a FastAPI backend + single-page frontend that gives users a natural language interface to their Google Calendar. Users authenticate with Google OAuth2, then chat with an AI agent that can:

- Check free/busy windows for any day
- Find the next available slot matching constraints (time of day, duration, day exclusions)
- Book a meeting with a two-step confirm flow
- List all events on a given day
- Search for specific events by name
- Handle relative scheduling ("after my standup", "next Tuesday morning")

In production, the UI switches to a **voice mode** — users speak their request, it is transcribed by Groq Whisper, processed by the agent, and the reply is spoken back using Groq TTS.

---

## Architecture & Request Flow

```
Browser
  │
  ├─[GET /]──────────────────────────────────► frontend/index.html (served by FastAPI)
  │
  ├─[GET /auth/login]────────────────────────► Google OAuth2 redirect
  │                                              ↓
  ├─[GET /auth/callback]─────────────────────► Token exchange → stored in server session
  │                                              (HttpOnly cookie set: ss_session)
  │
  ├─[POST /chat]─────────────────────────────► main.py
  │      body: {message, timezone}               │
  │                                              ├─ Validate session / refresh token
  │                                              ├─ Debounce check (1 s)
  │                                              └─► agent.run_agent()
  │                                                    │
  │                                                    ├─ Step 0: _fast_intent()
  │                                                    │   (zero LLM — set lookup for
  │                                                    │    obvious confirm/cancel words)
  │                                                    │
  │                                                    ├─ Step 1: intent.extract_intent()
  │                                                    │   LLM: llama-3.1-8b-instant
  │                                                    │   ~150 input tokens → JSON intent
  │                                                    │
  │                                                    └─ Step 2: _route()
  │                                                        (zero LLM — Python router)
  │                                                        │
  │                                                        ├─ check_availability
  │                                                        ├─ find_next_slot / find_slot_in_range
  │                                                        ├─ check_slot + set _pending
  │                                                        ├─ book_slot (on confirm)
  │                                                        ├─ get_events_for_day
  │                                                        └─ search_event
  │                                                              │
  │                                                              └─► Google Calendar API
  │
  ├─[POST /voice/transcribe]─────────────────► voice.py → Groq Whisper (STT)
  │      body: audio file (webm/wav/mp4)
  │
  └─[POST /voice/synthesize]─────────────────► voice.py → Groq TTS / Orpheus (TTS)
         body: {text}                             returns: audio/wav bytes
```

### Two-step Scheduling Pipeline — Token Efficiency

| Step | LLM calls | Approx. tokens |
|---|---|---|
| Step 0: fast intent (set lookup) | 0 | 0 |
| Step 1: intent extraction | 1 (8b model) | ~150 in / ~100 out |
| Step 2: routing + calendar ops | 0 | 0 |
| **Total per turn** | **1** | **~250** |
| Old tool-calling approach | 2–3 | ~4,000–7,200 |

---

## Project Structure

```
smart-scheduler/
├── backend/
│   ├── main.py           # FastAPI app, OAuth flow, /chat, /voice endpoints
│   ├── agent.py          # 2-step pipeline: fast-intent → extract → route → calendar
│   ├── intent.py         # LLM intent extraction (Step 1, minimal prompt ~150 tokens)
│   ├── calendar_tool.py  # Google Calendar API helpers (stateless, creds-injected)
│   ├── voice.py          # STT (Groq Whisper) + TTS (Groq Orpheus)
│   ├── memory.py         # Conversation state management (placeholder)
│   └── prompts.py        # Prompt constants (placeholder, pipeline moved to intent.py)
├── frontend/
│   └── index.html        # Single-page UI — text mode (dev) or voice mode (prod)
├── requirements.txt
├── render.yaml           # Render.com deployment config
└── .env                  # (not committed) secrets — see Environment Variables
```

---

## Features

### Scheduling Actions

| User says | Agent action |
|---|---|
| "Am I free tomorrow at 3 PM?" | `check_availability` → checks slot, shows conflicts or confirms |
| "Book a 30-min call with Sarah on Friday at 10 AM" | `book_explicit` → pending confirm → `book_slot` on confirm |
| "Find me a 1-hour slot next Tuesday morning" | `find_slot` → searches free windows with `not_before=08:00` |
| "What's on my calendar Wednesday?" | `list_events` → lists all events for that day |
| "When is my standup?" | `search_event` → searches by title within next 7 days |
| "Find a slot after my standup" | `find_slot` + anchor resolution → `not_before` = standup end time |
| "Find a slot next week, no Mondays" | `find_slot_in_range` + `exclude_days` constraint |
| "yes" / "go ahead" / "book it" | `confirm` → books the pending slot |
| "no" / "cancel" / "never mind" | `cancel` → clears pending slot |

### Multi-turn Context

The agent carries key fields (`title`, `date`, `duration_minutes`, time constraints) across turns so follow-up messages like "make it 45 minutes instead" work without repeating everything.

### Scheduling Constraints

- `not_before` / `not_after` — time window bounds ("before 6 PM", "after noon")
- `exclude_days` — skip specified weekdays in a date range search
- `anchor_event` + `anchor_relation` — schedule relative to an existing calendar event
- `anchor_offset_days` — schedule N days after a named event
- `buffer_minutes` — add buffer after an anchor event end time

### Voice Mode (Production)

- **STT**: Groq Whisper `whisper-large-v3-turbo` transcribes browser-recorded audio (webm / mp4 / wav)
- **TTS**: Groq Orpheus `canopylabs/orpheus-v1-english` (`troy` voice) synthesizes the agent reply as WAV
- Markdown stripped from TTS input so it doesn't read "asterisk" or "bullet point" aloud
- Time range dashes reformatted as "X to Y" for natural speech

### Security

- Google OAuth tokens stored **server-side only** — never sent to the browser
- Session IDs are cryptographically random (`secrets.token_urlsafe(32)`)
- HttpOnly cookies prevent JavaScript token theft
- OAuth `state` parameter validated on callback to prevent CSRF
- Sessions are fully isolated per user
- In development, credentials persist to `.credentials/<session_id>.json` so they survive `--reload`; this directory is not committed to version control

---

## Design Choices

### Why a 2-step pipeline instead of tool-calling?

Standard LLM tool-calling sends the full conversation history plus tool schemas on every turn — roughly 2,000–7,000 tokens per request. This project separates intent extraction from execution:

1. **Intent extraction** is a single small LLM call (~150 tokens) that outputs a structured JSON object with no side effects.
2. **Routing and calendar operations** are pure Python — zero LLM tokens.

This cuts cost and latency by ~10–20x per turn and makes the calendar logic deterministic and testable.

### Why Groq?

Groq provides a unified API for LLM inference, Whisper STT, and Orpheus TTS, so the entire AI stack runs through one key and one SDK. The inference speed (low-latency LPU hardware) also helps keep the voice loop responsive.

### Why server-side sessions?

Google OAuth tokens are never sent to the browser. A random session ID lives in an HttpOnly cookie; the actual credentials stay in the server's memory. This prevents token theft via XSS and avoids the risk of tokens being logged by a CDN, reverse proxy, or browser extension.

### Why a deterministic router over an LLM agent loop?

Calendar booking is a narrow, well-defined domain. A Python router with explicit branches for each action type is:
- **Predictable** — the same input always produces the same calendar operation
- **Debuggable** — every branch is visible code, not an LLM decision
- **Fast** — no second or third LLM call for tool selection

The LLM is only used for the one thing it does well: mapping natural language to a structured schema.

### Why a single-file frontend?

The UI is intentionally simple — a chat bubble interface in development, a voice orb in production. Keeping it in one `index.html` avoids a build step, keeps the deployment a single `uvicorn` process, and lets the FastAPI backend serve it directly with `FileResponse`.

---

## How the Agent Pipeline Works

### Step 0 — Fast Intent (zero tokens)

`_fast_intent()` checks if the message is a plain confirm or cancel word ("yes", "ok", "no", "cancel", "forget it", etc.) against a hard-coded word set. No LLM call is made.

### Step 1 — Intent Extraction (`intent.py`)

A minimal prompt (~150 tokens total) is sent to `llama-3.1-8b-instant`. The system prompt is a single sentence. The user message includes only: resolved date context, optional previous-turn context, a compact JSON schema, and the user's message.

The LLM returns a structured intent object:

```json
{
  "action": "find_slot",
  "title": "Team sync",
  "date": "2026-04-07",
  "duration_minutes": 60,
  "time": null,
  "date_range": null,
  "constraints": {
    "not_before": "09:00",
    "not_after": null,
    "exclude_days": []
  },
  "anchor_event": null,
  "anchor_relation": null,
  "anchor_offset_days": null,
  "buffer_minutes": null
}
```

Date references (today, tomorrow, named weekdays, "next week", "this week") are resolved to concrete ISO dates **before** the LLM call. Period words ("morning", "afternoon", "evening") map to `not_before` constraints rather than a literal `time` value.

### Step 2 — Deterministic Router (`agent.py`)

Zero LLM calls. The router executes Python calendar operations based on the intent `action` field:

- **`find_slot`** — calls `find_next_slot()` or `find_slot_in_range()`, resolves anchor events via `search_event()`, applies all constraints, sets a `_pending` entry in the session slot cache
- **`book_explicit`** — calls `check_slot()`, reports conflict info + alternative free windows if busy, sets `_pending` if free
- **`confirm`** — reads `_pending` from the slot cache and calls `book_slot()`
- **`cancel`** — clears `_pending` from the slot cache
- **`check_availability`** — calls `get_availability()` and formats free windows
- **`list_events`** — calls `get_events_for_day()` and formats the event list
- **`search_event`** — calls `search_event()` against the next 7 days

### Freebusy Cache

Each session maintains an in-memory `_slot_cache` keyed by `"date:tz_name"` with a 60-second TTL. This prevents redundant Google Calendar freebusy API calls when the same day is referenced multiple times in a single conversation (e.g., checking a slot then offering alternatives on the same day).

---

## API Reference

| Method | Path | Auth required | Description |
|---|---|---|---|
| `GET` | `/` | No | Serves `frontend/index.html` |
| `GET` | `/health` | No | Health check → `{"status":"ok"}` |
| `GET` | `/config` | No | Returns `{"mode":"development" or "production"}` |
| `GET` | `/auth/login` | No | Starts Google OAuth2 flow |
| `GET` | `/auth/callback` | No | OAuth callback — exchanges authorization code for tokens |
| `GET` | `/auth/status` | No | `{"authenticated": bool}` |
| `POST` | `/auth/logout` | Cookie | Clears server-side session and deletes cookie |
| `POST` | `/chat` | Cookie | Send message → `{"reply": str}` |
| `GET` | `/test-calendar` | Cookie | List tomorrow's free slots (debug endpoint) |
| `POST` | `/voice/transcribe` | Cookie | Upload audio file → `{"text": str}` |
| `POST` | `/voice/synthesize` | Cookie | `{"text": str}` → WAV audio bytes |

### `POST /chat`

```json
// Request body
{
  "message": "Book a 1-hour meeting called Project Review on Thursday at 10 AM",
  "timezone": "America/New_York"
}

// Response
{
  "reply": "Booking **Project Review** on Thursday, April 7 at 10:00–11:00 AM. Shall I go ahead?"
}
```

In development mode (`ENVIRONMENT=development`), the response also includes a `timing` object with per-step latency in milliseconds (LLM call, each calendar tool, total).

### `POST /voice/transcribe`

Accepts `multipart/form-data` with an `audio` file field. Any browser-recorded format (webm, mp4, wav) is accepted.

### `POST /voice/synthesize`

Returns raw `audio/wav` bytes. In development, the `X-Timing-Ms` response header contains synthesis latency.

---

## Environment Variables

Create a `.env` file in the project root (`smart-scheduler/.env`):

```env
GROQ_API_KEY=gsk_...
GOOGLE_CLIENT_ID=...apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=...
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/callback
ENVIRONMENT=development
SECRET_KEY=<random-string>
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key — used for LLM inference, Whisper STT, and Orpheus TTS |
| `GOOGLE_CLIENT_ID` | Yes | Google OAuth2 client ID from Google Cloud Console |
| `GOOGLE_CLIENT_SECRET` | Yes | Google OAuth2 client secret |
| `GOOGLE_REDIRECT_URI` | Yes | Must exactly match a redirect URI registered in Google Cloud Console |
| `ENVIRONMENT` | No | `development` (default) or `production`. Controls UI mode, logging level, and disk credential caching |
| `SECRET_KEY` | Yes (prod) | Application secret key |

### Google Cloud Console Setup

1. Create a project at [console.cloud.google.com](https://console.cloud.google.com)
2. Enable the **Google Calendar API**
3. Go to **APIs & Services → Credentials** and create **OAuth 2.0 credentials** (Web application type)
4. Add `http://localhost:8000/auth/callback` to *Authorized redirect URIs* for local development
5. Add `https://<your-app>.onrender.com/auth/callback` for production
6. While the app is in **Testing** mode, add your Google account under **Test users** on the OAuth consent screen

---

## Local Setup

**Requirements:** Python 3.11+

```bash
# 1. Enter the project directory
cd smart-scheduler

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env with your credentials (see Environment Variables above)

# 5. Start the development server
uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000` in your browser. Click **Connect Google Calendar**, complete the OAuth flow, then start chatting.

#### Example prompts

```
"Am I free tomorrow afternoon?"
"Book a 30-minute call on Friday at 2 PM"
"Find me a 1-hour slot next Tuesday, not before 9 AM"
"What's on my calendar this Thursday?"
"When is my standup?"
"Find a slot 1 hour after my standup"
"Schedule a meeting next week, skip Mondays"
"Last weekday of this month, find me a free hour"
```

---

## Deployment (Render)

The `render.yaml` file configures a Render Web Service with a single Python process.

**Steps:**

1. Push the repo to GitHub
2. On [render.com](https://render.com), create a new **Web Service** and link the repository — Render will detect `render.yaml` automatically
3. In the Render dashboard, set the environment variables that are marked `sync: false` in `render.yaml`:
   - `GROQ_API_KEY`
   - `GOOGLE_CLIENT_ID`
   - `GOOGLE_CLIENT_SECRET`
   - `GOOGLE_REDIRECT_URI` — set to `https://<your-app-name>.onrender.com/auth/callback`
   - `SECRET_KEY`
4. Update the Google Cloud Console OAuth consent screen to include the Render callback URL
5. Deploy — Render runs `pip install -r requirements.txt` then starts Uvicorn

The `ENVIRONMENT=production` value is hardcoded in `render.yaml`. In production: the frontend switches to voice UI, debug logging is suppressed, and the dev-only disk credential cache is disabled.

> **Note:** Render free-tier instances spin down after inactivity. The first request after a cold start will be slower due to DNS/TLS warmup, mitigated in code by a background warmup call on startup.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI + Uvicorn |
| LLM inference (primary) | Groq — `llama-3.3-70b-versatile` (fallback/complex turns) |
| LLM inference (intent) | Groq — `llama-3.1-8b-instant` (Step 1, ~150 tokens/turn) |
| Speech-to-text | Groq Whisper — `whisper-large-v3-turbo` |
| Text-to-speech | Groq Orpheus — `canopylabs/orpheus-v1-english` |
| Calendar integration | Google Calendar API v3 |
| Authentication | Google OAuth2 via `google-auth-oauthlib` |
| Frontend | Vanilla HTML / CSS / JavaScript (no framework) |
| Deployment | Render.com (`render.yaml`) |
| Python version | 3.11+ |
