# Architecture

## Components

![Components](diagrams/components.png)

```mermaid
flowchart LR
    user([User]) --> ui[Web page<br/>static HTML + JS]
    ui -- "JWT · JSON" --> api[FastAPI]
    subgraph backend [Backend · one container on Railway]
        api --> auth[Auth<br/>Argon2 + JWT]
        api --> chat[Conversations<br/>thread per user · rate limit]
        chat --> agent[LangChain create_agent<br/>LangGraph runtime]
        agent --> middleware[Middleware<br/>dynamic prompt · tool errors ·<br/>confirmation gate · call limit]
        agent --> tools[5 tools]
        tools --> service[BookingService<br/>every booking rule]
        middleware -- dry-run checks --> service
        api -- read-only REST --> service
        service --> db[(SQLite locally · Postgres on Railway<br/>PK room_id + slot_start)]
        agent --> memory[(InMemorySaver<br/>conversation state)]
    end
    agent -- Responses API --> openai[OpenAI gpt-6-luna]
    agent -. optional traces .-> langsmith[LangSmith]
```

## One message, from question to answer

![Sequence](diagrams/sequence.png)

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant UI as Web page
    participant API as FastAPI
    participant C as Conversations
    participant A as Agent
    participant M as OpenAI
    participant S as BookingService
    participant DB as Database

    U->>UI: "Book room B tomorrow 10–11:30 for 4, Interview"
    UI->>API: POST /chat/messages (Bearer JWT)
    API->>API: JWT → CurrentUser · rate limit
    API->>C: send_message(user, conversation, text)
    C->>A: invoke(message, thread = user:conversation, context = user)
    A->>A: dynamic prompt: rooms, office time, rules
    A->>M: messages + tool schemas (no user field)
    M-->>A: create_booking(room, start, end, title, attendees)
    A->>S: when(): check_create, a dry run
    S->>DB: room, overlapping bookings
    S-->>A: valid
    A-->>C: interrupt with the summary
    C-->>API: pending_actions
    API-->>UI: confirmation card
    U->>UI: Confirm
    UI->>API: POST /chat/decisions {approve: true}
    API->>C: decide(user, conversation, approve)
    C->>A: resume with approve
    A->>S: create(user from context, request)
    S->>DB: INSERT booking + one row per slot
    DB-->>S: ok, or a primary key violation → SLOT_TAKEN
    S-->>A: {"ok": true, "booking": ...}
    A->>M: tool result
    M-->>A: "Done, room B is yours..."
    A-->>C: final message
    C-->>API: reply
    API-->>UI: reply · the page refreshes the schedule
```

## Middleware order

| Middleware | Hook | What it does |
|---|---|---|
| `system_prompt` (`@dynamic_prompt`) | before each model call | rooms, capacities, office date and time, rules, the user's name |
| `booking_errors_as_tool_results` (`@wrap_tool_call`) | around each tool | `BookingError` → structured tool result; any other exception → `INTERNAL_ERROR`, logged |
| `HumanInTheLoopMiddleware` | after each model call | pauses `create_booking` / `cancel_booking` only when a dry run on the schema-validated arguments passes |
| `ModelCallLimitMiddleware(run_limit=6)` | per user message | stops a runaway loop |

## Where each guarantee lives

| Guarantee | Enforced by |
|---|---|
| No double booking, even concurrently | primary key `(room_id, slot_start)` |
| Capacity, 30-minute alignment, 3 hours, business hours, title | `BookingService.check_create` |
| Only the owner cancels | `BookingService.check_cancel` with the user from runtime context |
| Nobody acts as another user | the user is never a tool argument; threads are keyed by user on the server |
| No write without the user's consent | `HumanInTheLoopMiddleware` + `/chat/decisions` |
| No cross-user prompt injection through titles | schedules hide other users' titles |
