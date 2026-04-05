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
- Handle relative scheduling ("after my standup", "last weekday of next month", "end of next quarter")

In **development**, the UI shows a standard text chat interface. In **production**, the entire interface becomes voice-driven — no typing required. The user speaks, the browser detects when they stop talking, the audio is transcribed, the agent replies, and the reply is spoken back.

### What a conversation looks like

```
User:  Find me a free hour on the last weekday of next month.
Agent: Found a 60 min slot: Friday, May 29 at 10 AM–11 AM. Shall I book it as Meeting?

User:  Actually make it a standup, 30 minutes.
Agent: Found a 30 min slot: Friday, May 29 at 10 AM–10:30 AM. Shall I book it as Standup?

User:  Yes.
Agent: Done! Standup is on your calendar for Friday, May 29 at 10 AM–10:30 AM.
```

The agent remembers `date`, `time`, `duration`, and `title` across turns — "actually make it 30 minutes" works without repeating the date.

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
  │                                                    │   zero LLM — word-set lookup
  │                                                    │   for confirm / cancel
  │                                                    │
  │                                                    ├─ Step 1: intent.extract_intent()
  │                                                    │   LLM: llama-3.1-8b-instant
  │                                                    │   input: today, tomorrow, prev
  │                                                    │   context, schema, message
  │                                                    │
  │                                                    │   [simple date] → JSON intent
  │                                                    │
  │                                                    │   [relative date, e.g. "last
  │                                                    │    weekday of next month"]:
  │                                                    │   LLM calls get_date_map tool
  │                                                    │        ↓
  │                                                    │   Python returns date map
  │                                                    │        ↓
  │                                                    │   LLM reads map → JSON intent
  │                                                    │
  │                                                    └─ Step 2: _route()
  │                                                        zero LLM — Python router
  │                                                        │
  │                                                        ├─ find_next_slot
  │                                                        ├─ find_slot_in_range
  │                                                        ├─ check_slot + set _pending
  │                                                        ├─ book_slot (on confirm)
  │                                                        ├─ get_events_for_day
  │                                                        ├─ search_event (incl. anchor)
  │                                                        └─ get_availability
  │                                                              │
  │                                                              └─► Google Calendar API
  │
  │  ── Voice pipeline (production only) ──────────────────────────────────────────
  │
  │  [Browser VAD loop — entirely client-side]
  │  @ricky0123/vad-web + onnxruntime-web (WASM)
  │  Silero model detects speech → emits WAV segment
  │        │
  ├─[POST /voice/transcribe]─────────────────► voice.py → Groq Whisper STT
  │      body: WAV blob                          returns: {"text": "..."}
  │                                              (browser then calls /chat with the text)
  │
  └─[POST /voice/synthesize]─────────────────► voice.py → Groq Orpheus TTS
         body: {text}                             returns: audio/wav bytes
                                                  (browser plays via AudioContext)
```

### Two-step Scheduling Pipeline — Token Efficiency

| Step | LLM calls | Approx. tokens |
|---|---|---|
| Step 0: fast intent (set lookup) | 0 | 0 |
| Step 1: intent extraction — simple date | 1 (`8b` model) | ~150 in / ~100 out |
| Step 1: intent extraction — complex relative date | 2 (`8b` model) | ~150 + map + ~100 out |
| Step 2: routing + calendar ops | 0 | 0 |
| Step 3: NL fallback (unknown intent only) | 0 or 1 | ~80 out |
| **Typical turn (known intent, simple date)** | **1** | **~250** |
| **Turn with relative date** | **2** | **~500** |
| **Worst case (unknown intent)** | **2** | **~330** |
| Old tool-calling approach | 2–3 | ~4,000–7,200 |

---

## Project Structure

```
smart-scheduler/
├── backend/
│   ├── main.py           # FastAPI app, OAuth flow, /chat, /voice, /session endpoints
│   ├── agent.py          # 2-step pipeline: fast-intent → extract → route → calendar
│   ├── intent.py         # LLM intent extraction with get_date_map tool for relative dates
│   ├── calendar_tool.py  # Google Calendar API helpers (stateless, creds-injected)
│   ├── voice.py          # STT (Groq Whisper) + TTS (Groq Orpheus)
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

The agent carries key fields across turns — `title`, `date`, `duration_minutes`, `time`, `not_before`, `not_after` — so follow-up messages work without the user repeating context.

The rule is **null-coalescing**: if the LLM returned `null` for a field, the previous turn's value wins. If the LLM returned a non-null value, the new value wins (user said something new).

An exact `time` value (e.g. `14:00`) takes precedence over `not_before`/`not_after` constraints from a previous turn — they would conflict, so the constraints are dropped when an explicit time is carried forward.

**Example:**
```
Turn 1: "Find a slot Friday afternoon"
  → intent: date=2026-04-10, not_before=12:00, duration=60
  → "Found 60 min: Friday at 1 PM–2 PM. Book as Meeting?"

Turn 2: "Change duration to 90 minutes"
  → LLM returns: duration=90, date=null, not_before=null
  → carryover fills: date=2026-04-10, not_before=12:00
  → "Found 90 min: Friday at 1 PM–2:30 PM. Book as Meeting?"

Turn 3: "Make it 2 PM"
  → LLM returns: time=14:00, date=null, duration=null
  → carryover fills: date=2026-04-10, duration=90
  → not_before dropped (explicit time supersedes)
  → "Booking Meeting on Friday at 2 PM–3:30 PM. Shall I go ahead?"
```

### Session Reset

A **↻ New Conversation** button in the UI calls `POST /session/reset`, which clears the slot cache and conversation context for the current session without affecting Google authentication. The user can start a fresh scheduling conversation without a full page reload.

### Scheduling Constraints

- `not_before` / `not_after` — time window bounds ("before 6 PM", "after noon", "morning", "afternoon")
- `exclude_days` — skip specific weekdays across a date range ("next week, no Mondays or Fridays")
- `anchor_event` + `anchor_relation` — schedule relative to an existing calendar event ("after my standup", "before the all-hands")
- `anchor_offset_days` — schedule a given number of days after a named event ("a day or two after the kickoff")
- `buffer_minutes` — add a buffer gap after the anchor event ends before placing the new meeting

When an anchor event is referenced, the router calls `search_event()` to find it on the calendar before computing the time constraint. If the event is not found or a network error occurs, the agent replies with a clear error rather than silently scheduling at the wrong time.

### Confirmation Flow

No meeting is ever booked in a single step. The agent always:
1. Proposes a slot ("Found 60 min: Friday at 10 AM–11 AM. Shall I book it as Meeting?")
2. Waits for an explicit confirm ("yes", "go ahead", "book it", "sounds good", etc.)
3. Only then calls `book_slot()` against the Google Calendar API

The pending slot is stored in the session's `_slot_cache` under `_pending`. Saying "no", "cancel", or "never mind" clears it. If the user says something ambiguous after a proposal, the agent reminds them of the pending booking rather than starting over.

### Voice Mode (Production)

#### Why vad-web instead of LiveKit (or any server-side voice framework)?

The obvious choice for a voice assistant is a server-side media framework like LiveKit, which streams audio from the browser to a server process that runs a VAD model, performs STT, and streams TTS back. That works well on hardware where memory is not a constraint.

This project targets **Render's free tier** — 512 MB of RAM for the entire Python process. A server-side VAD model (Silero via PyTorch) would consume ~300–400 MB on its own, leaving no headroom for FastAPI, the Groq client, or Google Calendar calls. LiveKit itself also requires a separate relay/SFU server.

The solution: move VAD entirely into the browser. `@ricky0123/vad-web` runs the same Silero ONNX model client-side via `onnxruntime-web` (WebAssembly). The server never sees a raw audio stream — it only receives a finished WAV segment after the browser has already detected that the user stopped speaking. This means:

- **Zero server RAM** for voice activity detection
- **No persistent connection** — each speech segment is a standard HTTP POST
- **Works on free hosting** — the Python process stays well within 512 MB
- **No LiveKit relay server** needed — the browser handles the real-time audio entirely

The trade-off: the browser needs to load a ~1.5 MB WASM binary and the Silero ONNX model (~1 MB) on first load. This is a one-time cost and acceptable for a scheduling assistant.

#### How browser-side VAD works

1. User clicks **🎤 Start** → `AudioContext` is created (browser requires a user gesture before any audio API). This same context is reused for TTS playback.
2. `MicVAD` from `@ricky0123/vad-web` opens the microphone and continuously feeds audio frames to the Silero model running in WASM.
3. When the model detects speech onset → button turns red / **🔴 Hearing…**
4. When the model detects speech end (configurable silence threshold) → `onSpeechEnd` fires with a `Float32Array` of PCM audio.
5. The browser converts the PCM to a WAV blob and POSTs it to `/voice/transcribe`.
6. The transcript is passed to `/chat` for the agent pipeline.
7. The agent reply text is POSTed to `/voice/synthesize` → WAV bytes played via `AudioContext.decodeAudioData()`.

#### Key VAD configuration

| Parameter | Value | Effect |
|---|---|---|
| `positiveSpeechThreshold` | `0.6` | Confidence needed to declare speech started |
| `negativeSpeechThreshold` | `0.35` | Confidence below which silence is declared |
| `minSpeechFrames` | `4` | Minimum frames to count as speech (filters clicks/coughs) |
| ORT `numThreads` | `1` | Single WASM thread — avoids SharedArrayBuffer requirement on some hosts |
| ORT `logLevel` | `error` | Suppresses the Silero model's verbose optimization logs |

#### TTS reliability

- The Groq TTS endpoint occasionally drops connections under brief server load spikes. `speak_text()` retries up to 3 times with 0.5 s and 1.0 s back-off before failing.
- HTTP 4xx responses (bad input, quota) fail immediately without retrying.
- Markdown is stripped before synthesis so the TTS doesn't read "asterisk" or "hash" aloud.
- Time ranges like "9 AM–10 AM" are rewritten to "9 AM to 10 AM" for natural-sounding speech.

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

### Why the `get_date_map` tool for relative dates?

Early versions tried to resolve relative date phrases ("last weekday of next week") in Python using regex rules before the LLM ever saw them. This broke in two ways: regex can't enumerate every phrase a user might say, and adding a new phrasing required editing code.

The current design inverts responsibility:
- Python handles what it's good at: arithmetic (given a range, return every date with its label)
- The LLM handles what it's good at: understanding natural language ("second Tuesday", "last weekday", "end of next quarter")

The `get_date_map` tool takes an optional `start` and `end` date and returns `{"YYYY-MM-DD": "Weekday D Mon YYYY"}` for every day in the range. The LLM reads the map and picks the right date. No new code is needed when a user says something unexpected — the LLM just requests an appropriate range and reads it.

For simple turns ("tomorrow", "Friday") the LLM resolves the date directly from `today`/`tomorrow` without calling the tool, so there's no latency penalty for the common case.

### Why vad-web instead of LiveKit (or any server-side voice framework)?

See the [Voice Mode](#voice-mode-production) section for the full explanation. Short version: this project targets Render's free tier (512 MB RAM). A server-side VAD+PyTorch stack consumes 300–400 MB minimum — not viable. Running the same Silero model client-side in WebAssembly uses zero server memory and needs no separate relay server.

### Why Groq?

Groq provides a unified API for LLM inference, Whisper STT, and Orpheus TTS, so the entire AI stack runs through one key and one SDK. The inference speed (low-latency LPU hardware) also keeps the voice loop responsive — typically under 500 ms for STT + agent + TTS combined.

### Why server-side sessions?

Google OAuth tokens are never sent to the browser. A random session ID lives in an HttpOnly cookie; the actual credentials stay in the server's memory. This prevents token theft via XSS and avoids the risk of tokens being logged by a CDN, reverse proxy, or browser extension.

### Why a deterministic router over an LLM agent loop?

Calendar booking is a narrow, well-defined domain. A Python router with explicit branches for each action type is:
- **Predictable** — the same input always produces the same calendar operation
- **Debuggable** — every branch is visible code, not an LLM decision
- **Fast** — no second or third LLM call for tool selection

The LLM is only used for the one thing it does well: mapping natural language to a structured schema.

### Why a single-file frontend?

The UI is intentionally simple — a chat bubble interface in development, a voice orb in production. Keeping it in one `index.html` avoids a build step, keeps the deployment a single `uvicorn` process, and lets the FastAPI backend serve it directly with `FileResponse`. It also means voice features ship without a CDN or asset pipeline.

---

## How the Agent Pipeline Works

### Step 0 — Fast Intent (zero tokens)

`_fast_intent()` checks the message against two hard-coded word sets — `_CONFIRM_WORDS` and `_CANCEL_WORDS` — before any LLM call is made. Words like "yes", "yep", "ye", "ya", "sounds good", "go for it" all map to `confirm`. "No", "nah", "forget it", "never mind" map to `cancel`. This handles the most common turn type (saying yes or no to a proposal) with zero latency and zero cost.

### Step 1 — Intent Extraction (`intent.py`)

A minimal prompt is sent to `llama-3.1-8b-instant`. The prompt contains only: `today` + `tomorrow` ISO dates, optional previous-turn context (compact `key=val` string), a compact JSON schema, and the user's message.

The LLM is instructed to return `date=null` if the user did not mention a date — this is critical for multi-turn carryover to work correctly (a non-null value from the LLM means the user said something new; `null` means inherit from context).

**Simple date turn — 1 LLM call:**
```
Prompt → "Find a slot tomorrow afternoon"  
LLM   → {"action":"find_slot","date":"2026-04-06","constraints":{"not_before":"12:00"},...}
```

**Relative date turn — 2 LLM calls (1 tool call in between):**
```
Prompt  → "Last weekday of next month"
LLM     → calls get_date_map({"start":"2026-05-01","end":"2026-05-31"})
Python  → {"2026-05-01":"Friday 1 May 2026", ..., "2026-05-29":"Friday 29 May 2026", ...}
LLM     → {"action":"find_slot","date":"2026-05-29",...}
```

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

Period words ("morning", "afternoon", "evening") map to `not_before` constraints rather than a literal `time` value. `time` is reserved for exact start times like "at 2 PM".

### Step 2 — Deterministic Router (`agent.py`)

Zero LLM calls. The router executes Python calendar operations based on the intent `action` field:

- **`find_slot`** — calls `find_next_slot()` or `find_slot_in_range()`, resolves anchor events via `search_event()`, applies all constraints, sets a `_pending` entry in the session slot cache
- **`book_explicit`** — calls `check_slot()`, reports conflict info + alternative free windows if busy, sets `_pending` if free
- **`confirm`** — reads `_pending` from the slot cache and calls `book_slot()`
- **`cancel`** — clears `_pending` from the slot cache
- **`check_availability`** — calls `get_availability()` and formats free windows
- **`list_events`** — calls `get_events_for_day()` and formats the event list
- **`search_event`** — calls `search_event()` against the next 7 days

### Step 3 — NL Fallback (unknown intent only)

If the router returns `None` (intent was `"unknown"`), the agent first checks whether there is a pending booking in the session. If so, it reminds the user of it ("Did you want to confirm booking **X** at …?") with no LLM call. Only when there is no pending booking does it make a second LLM call — ~80 output tokens — using a minimal system prompt constrained to scheduling topics.

### Freebusy Cache

Each session maintains an in-memory `_slot_cache` keyed by `"date:tz_name"` with a **5-minute (300-second) TTL**. This prevents redundant Google Calendar freebusy API calls when the same day is referenced multiple times in a single conversation (e.g., checking a slot then offering alternatives on the same day).

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
| `POST` | `/session/reset` | Cookie | Clears slot cache and conversation context; preserves Google auth |

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
# Basic availability
"Am I free tomorrow afternoon?"
"What's on my calendar this Thursday?"
"Am I free from 1 PM to 2 PM on Friday?"

# Booking with constraints
"Book a 30-minute call on Friday at 2 PM"
"Find me a 1-hour slot next Tuesday, not before 9 AM"
"Schedule a meeting next week, skip Mondays"
"Find a slot before 6 PM on Wednesday"

# Anchor-relative scheduling
"When is my standup?"
"Find a slot 1 hour after my standup"
"Schedule something the day after the project kickoff"
"Find a slot before the all-hands on Friday"

# Complex relative dates (handled by get_date_map tool)
"Last weekday of next month — find me a free hour"
"Book something on the second Tuesday of May"
"Find a slot at end of next week"
"Schedule a 45-minute call in three weeks, afternoon"

# Multi-turn refinement
"Find a slot Friday afternoon"  →  "Make it 90 minutes"  →  "Change to 2 PM"  →  "Yes"
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
| LLM inference (primary) | Groq — `llama-3.3-70b-versatile` (Step 1 intent extraction + Step 3 NL fallback) |
| LLM inference (rate-limit fallback) | Groq — `llama-3.1-8b-instant` (used when 70b is rate-limited) |
| Speech-to-text | Groq Whisper — `whisper-large-v3-turbo` |
| Text-to-speech | Groq Orpheus — `canopylabs/orpheus-v1-english` |
| Calendar integration | Google Calendar API v3 |
| Authentication | Google OAuth2 via `google-auth-oauthlib` |
| Voice activity detection | [`@ricky0123/vad-web`](https://github.com/ricky0123/vad) `0.0.22` — Silero ONNX model, runs in-browser |
| ONNX runtime (browser) | `onnxruntime-web` `1.14.0` — WASM execution provider for the VAD model |
| Frontend | Vanilla HTML / CSS / JavaScript (no framework) |
| Deployment | Render.com (`render.yaml`) |
| Python version | 3.11+ |
