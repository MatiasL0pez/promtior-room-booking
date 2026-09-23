# Room Booking Assistant — Design Spec

Date: 2026-09-22 · Status: approved in conversation, pending written review

## 1. Goal

A conversational assistant that books meeting rooms at Cubo Itaú through tool calling, for the
Promtior AI Engineer technical challenge. Success means: every booking rule is enforced by code and
the database (never by the prompt), the assistant takes the right path on every request (answer,
clarify, refuse, fail with a clear reason, or act after confirmation), and an evaluator can read,
run, test and use the deployed solution without asking anything.

## 2. Requirements traceability

| Requirement (challenge PDF) | Where it is satisfied |
|---|---|
| Five rooms A–E, each with a capacity | `rooms` table, seeded (§4) |
| 30-minute slots | `[start, end)` aligned to :00/:30 (§4) |
| Attendees must not exceed capacity | `CAPACITY_EXCEEDED` rule (§4) |
| No double bookings, no overlaps | `booking_slots` primary key `(room_id, slot_start)` (§4) |
| Only contiguous slots, max 3 hours | A booking is one aligned range; `TOO_LONG` rule (§4) |
| Every booking has a title | `TITLE_REQUIRED` rule (§4) |
| Login for User1 / User2 with the given password | `POST /auth/login`, JWT (§6) |
| Create a booking for the logged-in user | `create_booking` tool; user comes from runtime context (§5) |
| List available rooms for a time range | `list_available_rooms` tool (§5) |
| Room schedule, available vs occupied | `get_room_schedule` tool (§5) |
| Cancel own bookings only | `cancel_booking` tool + `NOT_OWNER` rule (§4, §5) |
| Jupyter notebook explaining the technologies | `doc/walkthrough.ipynb` (§9) |
| Project overview + component diagram in `/doc` | `doc/README.md`, `doc/architecture.md` (§9) |
| Cloud deployment | Railway, Dockerfile + managed Postgres (§10) |
| Public GitHub repository | this repo |

## 3. Assumptions

Not specified by the challenge; documented and configurable:

- Capacities: A=4, B=6, C=8, D=12, E=20.
- Bookable hours: 08:00–20:00, any day of the week.
- Single time zone: `America/Montevideo`. Tools speak local ISO 8601 (`2026-09-23T10:00`); the
  database stores UTC.
- Conversations are working memory (in-process); bookings are the durable record.

## 4. Domain

### Tables

```
users          id · username UNIQUE · password_hash
rooms          id ('A'..'E') · capacity
bookings       id · room_id · user_id · title · attendees · start_at · end_at · created_at · cancelled_at
booking_slots  room_id · slot_start · booking_id          PRIMARY KEY (room_id, slot_start)
```

- A booking is a half-open range `[start, end)` aligned to 30 minutes, so its slots are contiguous
  by construction and `10:00–11:30` does not conflict with `11:30–12:00`.
- Creating a booking inserts the booking and one `booking_slots` row per 30-minute slot in one
  transaction. The primary key makes a double booking impossible even under concurrent requests.
- Cancelling sets `cancelled_at` and deletes the booking's slot rows in the same transaction: the
  slots become free and the booking stays in history.

### Rules (BookingService)

`BookingService` holds every rule, knows nothing about the LLM, and receives the clock as a
dependency. Each violation is a `BookingError` with a code and the data needed to explain it.

| Rule | Error code | Extra data |
|---|---|---|
| Room exists | `ROOM_NOT_FOUND` | valid rooms |
| End after start | `INVALID_TIME_RANGE` | |
| Start and end on :00 or :30 | `NOT_ALIGNED` | |
| Duration ≤ 3 hours | `TOO_LONG` | max duration |
| Start not in the past | `IN_THE_PAST` | current time |
| Inside bookable hours | `OUTSIDE_BUSINESS_HOURS` | opening hours |
| 1 ≤ attendees ≤ capacity | `CAPACITY_EXCEEDED` / `INVALID_ATTENDEES` | capacity |
| Title not blank, ≤ 120 chars | `TITLE_REQUIRED` / `TITLE_TOO_LONG` | |
| No overlap with an active booking | `SLOT_TAKEN` | conflicting range |
| Booking exists | `BOOKING_NOT_FOUND` | |
| Only the owner cancels | `NOT_OWNER` | |
| Not already cancelled | `ALREADY_CANCELLED` | |
| Not already started | `ALREADY_STARTED` | |
| Query range ≤ 7 days | `RANGE_TOO_LARGE` | max range |

Operations: `check_create`, `create`, `check_cancel`, `cancel`, `available_rooms`,
`room_schedule`, `bookings_of`. The `check_*` operations run the same validations without writing.

### Data minimization

In a room schedule, another user's booking appears only as `occupied`, without its title. Text
written by one user never reaches another user's LLM context, which closes a cross-user prompt
injection path. Own bookings include id and title.

## 5. Agent

Built with LangChain `create_agent` (LangGraph runtime) and an OpenAI chat model chosen by the
`OPENAI_MODEL` environment variable.

### Tools

| Tool | Kind | Returns |
|---|---|---|
| `list_available_rooms(start, end, attendees?)` | read | rooms free for the whole range, with capacity |
| `get_room_schedule(room, start, end)` | read | compact ranges: `10:00–11:30 occupied (mine, id 7, "Interview") · 11:30–20:00 free` |
| `list_my_bookings()` | read | the user's active future bookings with ids |
| `create_booking(room, start, end, title, attendees)` | write, confirmed | the created booking |
| `cancel_booking(booking_id)` | write, confirmed | the cancelled booking |

- The logged-in user reaches the tools through `ToolRuntime` context (`context_schema`). It is not
  a tool argument, so it is absent from the schema the model sees and cannot be forged by a prompt.
- Tools return JSON: `{"ok": true, ...}` or `{"ok": false, "error": "<CODE>", "message": ..., ...data}`.
  Domain errors never raise to the agent. Unexpected exceptions are turned into `INTERNAL_ERROR` by
  a `wrap_tool_call` middleware and logged.

### Confirmation of writes

`HumanInTheLoopMiddleware` guards `create_booking` and `cancel_booking` with
`allowed_decisions = ["approve", "reject"]` and a `when` predicate that runs `check_create` /
`check_cancel`:

- Invalid request → no pause; the tool runs, returns the error, nothing is written, the model
  explains and offers alternatives.
- Valid request → pause; the UI shows a confirmation card built by the `description` callable
  ("Room B · Wed 23 Sep, 10:00–11:30 · Interview with John · 4 attendees").
- Approve → the tool writes. A slot lost to a concurrent booking in between returns `SLOT_TAKEN`.
- Reject → the tool does not run.
- Typing a message while a confirmation is pending counts as a reject whose message is the user's
  text ("make it 11 instead"), so the model adjusts and proposes again.
- The model may make several tool calls in one step ("book D every day this week"): the pause lists
  every write that would succeed on one card, and one decision applies to all of them.

### System prompt

Rebuilt on every turn with: the assistant's sole purpose; rooms and capacities; the current date,
time and weekday in Montevideo; bookable hours and rules; instructions to ask for any missing
field instead of guessing, to find booking ids with `list_my_bookings`, not to ask for
confirmation in text (the system pauses on its own), to reply in the user's language, in short
plain text, and to decline anything outside room booking.

### State and limits

- Checkpointer: `InMemorySaver`. Thread id = `"{user_id}:{conversation_id}"`, built on the server,
  so a user can never resume another user's thread.
- At most 6 model calls per user message (`ModelCallLimitMiddleware`); message length ≤ 1000
  characters.

## 6. API, authentication, UI

- `POST /auth/login` → JWT (HS256, 8 h). Users seeded with hashed passwords.
- `POST /chat/messages` `{conversation_id?, message}` and `POST /chat/decisions`
  `{conversation_id, approve}` → `{conversation_id, reply, pending_actions: [{tool, summary}]}`.
  A decision with no pending action → 409.
- `GET /auth/me` → the logged-in user, so the UI can check a stored token.
- `GET /rooms`, `GET /rooms/{id}/schedule?start&end`, `GET /bookings/mine` → read-only REST over
  the same service, used by the schedule panel.
- `GET /health`.
- UI: one static page served by FastAPI, vanilla HTML/JS, no build step. Login form, chat panel,
  confirmation card with Confirm / Cancel, and a day grid (rooms × 08:00–20:00) that refreshes after
  every assistant turn and highlights the user's bookings. Assistant text is rendered as text, never
  as HTML.

## 7. Quality

### Tests (pytest, no API key, seconds)

- Domain: every rule in §4 with one passing and one failing case, fixed clock, SQLite in memory;
  a race test where two transactions take the same slot and exactly one wins.
- API: login ok / wrong password / 401 without token; REST endpoints; User1 cannot cancel a
  booking of User2; decision without pending action → 409.
- Agent wiring with a fake chat model that emits scripted tool calls: pause only on valid writes,
  approve writes, reject does not, `user_id` absent from every tool schema.

### Evals (`pytest -m eval`, real model, excluded by default)

About ten cases covering the five response paths. Assertions are on tool calls and final database
state, never on wording.

| Path | Cases |
|---|---|
| Answer | free rooms for 8 people tomorrow 15:00–16:00; schedule of room C |
| Clarify | "book me a room tomorrow" (no title, no attendees) → asks, writes nothing |
| Refuse | "write me a poem"; "cancel User2's booking" |
| Structured failure | 30 people in room A → nothing written, capacity explained |
| Act | pause carries correct args and approve persists it; relative dates ("next Friday"); correction next turn ("make it 6 people"); replies in the user's language |

### Cost and abuse controls

The repository is public and the password is in the challenge, so anyone can spend the OpenAI
credit: per-user limit of 30 messages per 10 minutes, the step and length limits above, and a
dedicated OpenAI project with a monthly budget (configured in the OpenAI dashboard).

### Observability

LangSmith tracing turned on through environment variables only (`LANGSMITH_TRACING`,
`LANGSMITH_API_KEY`). Application logs record tool calls, their error codes and the thread id.

## 8. Repository layout

```
app/
  config.py      settings from environment
  db.py          engine, session, create tables, seed
  models.py      SQLAlchemy tables
  booking.py     BookingService, BookingError
  auth.py        login, JWT, current user
  agent.py       tools, prompt, middleware, create_agent
  chat.py        conversations (message, decision, reply) and the per-user rate limit
  main.py        FastAPI app and routes
  static/index.html
tests/           support.py · test_config.py · test_db.py · test_booking.py · test_api.py · test_agent.py · test_chat.py
evals/           test_evals.py
doc/             README.md · architecture.md · walkthrough.ipynb · process/
Dockerfile · pyproject.toml · uv.lock · .env.example · .gitattributes · .github/workflows/ci.yml · README.md
```

## 9. Documentation deliverables (`/doc`)

- `README.md` — project overview: approach, key decisions, challenges and how they were solved,
  assumptions, limitations, next steps. Drafted, then rewritten in the author's own words.
- `architecture.md` — component diagram and the sequence of one message from question to answer,
  in Mermaid (rendered by GitHub) and exported to PNG.
- `walkthrough.ipynb` — each technology with real code from the repo: rules in action, the slot
  race, the tool schema without `user_id`, a real conversation with pause and approval, and the eval
  summary. Committed executed so it reads without an API key.
- `process/` — this spec and the implementation plan.

## 10. Deployment and tooling

- `uv` + `pyproject.toml`, `ruff` (lint and format), GitHub Actions running lint and tests (not
  evals), badge in the root README. `.gitattributes` forces LF.
- `Dockerfile`; Railway service from the GitHub repo plus Railway Postgres. Variables:
  `OPENAI_API_KEY`, `OPENAI_MODEL`, `DATABASE_URL` (normalized to the psycopg driver), `JWT_SECRET`,
  optional LangSmith. Tables and seed created at startup, idempotent. Healthcheck `/health`.

## 11. Out of scope

Token streaming, multiple replicas and shared conversation state (a Postgres checkpointer is the
next step), per-user time zones, database migrations (Alembic is the next step), write endpoints in
the REST API, recurring bookings.

## 12. Verified before coding (2026-09-22)

Checked against the documentation of the day and by running probes against the installed
libraries (langchain 1.4.2, langgraph 1.2.12, langchain-openai 1.6.3, fastapi 0.141.1):

1. The `when` predicate receives a `ToolCallRequest` whose `runtime.context` is the user. It runs
   again when the graph resumes, so a booking that became invalid while waiting is not paused
   again: the tool runs, fails validation and writes nothing.
2. `@dynamic_prompt` middleware rebuilds the system prompt on every model call.
3. `GenericFakeChatModel` lacks `bind_tools`; a subclass that returns itself scripts tool calls.
4. Default model `gpt-6-luna` (OpenAI models page, 2026-09-22): function calling, reasoning effort
   levels, USD 0.10 / 0.50 per million input / output tokens. Called through the Responses API.
5. `invoke(..., version="v2")` returns `.value` and `.interrupts`; `get_state(config).interrupts`
   tells whether a confirmation is pending; `reject` with a `message` reaches the model as the
   user's reason.
6. FastAPI recommends `pwdlib[argon2]` and `PyJWT`. `OAuth2PasswordRequestForm` needs
   `python-multipart`; `tzdata` is needed for time zones on Windows and slim images.
