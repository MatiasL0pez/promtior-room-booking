# Project overview

## How I approached it

I started from the rules, not from the chatbot. The challenge is really two systems: a booking
system with strict constraints and a conversational layer on top of it. If the rules live in the
prompt, the model eventually breaks them; so every rule lives in `BookingService` and in the
database, and the agent is one more client of that service, exactly like the REST API.

Then I defined what "behaving well" means for the assistant: for every request it has to take
the right path — answer, ask for what is missing, refuse, explain a failure, or act — and it may
act only with the user's consent. The design follows from those two decisions.

## Implementation logic

1. **Domain first.** `BookingService` validates room, 30-minute alignment, duration, past, business
   hours, capacity, title and overlap, and returns a code plus the data needed to explain the
   problem (`CAPACITY_EXCEEDED` carries the capacity, `SLOT_TAKEN` the conflicting range).
2. **The database as the last line.** Each booking stores one row per 30-minute slot with primary
   key `(room_id, slot_start)`. Two requests for the same slot at the same time cannot both succeed.
3. **Identity outside the model.** The JWT becomes a `CurrentUser` that reaches the tools through
   LangChain's `ToolRuntime` context. It is not a tool argument, so no prompt can book or cancel
   as someone else.
4. **Consent before writes.** `HumanInTheLoopMiddleware` pauses `create_booking` and
   `cancel_booking`, but only when a dry run (`check_create` / `check_cancel`) says the action will
   succeed. The user never confirms a booking that then fails.
5. **Evals on behavior.** Unit tests script the model; evals run the real model several times per
   case (`EVAL_REPEAT`) and assert on the tools it called, the final database state and the reply
   language, never on wording, so a behavior that fails one time in ten shows up as a rate.

## Key decisions

| Decision | Why | Alternative I rejected |
|---|---|---|
| LangChain `create_agent` on LangGraph | confirmation, runtime context and checkpointing come built in | a hand-written OpenAI loop (would rebuild all three) or a custom `StateGraph` (same graph, more code) |
| Slot rows with a primary key | a literal translation of "a slot can only be held by one booking", portable to SQLite and Postgres | Postgres exclusion constraint (ties local development and tests to Postgres) |
| Pause only valid writes (`when` predicate) | a confirmation of something that will fail is noise | pausing every write |
| Other users' titles hidden in schedules | least privilege, and it closes a cross-user prompt injection path | showing every title |
| Typing while a card is pending = reject with that text | "make it 6 people" should just work | forcing a click first |
| `gpt-6-luna` through the Responses API | cheapest current model with function calling; switchable by an environment variable | a larger model without evidence that it is needed |
| Per-user message limit | the repository and the password are public | no limit |
| One language rule as the first instruction, no office location in the prompt | measured: the model stopped answering in the office's language | a longer language rule, or naming the office's language in it (made it worse) |

## Challenges and how I solved them

- **Confirming bookings that would fail.** `HumanInTheLoopMiddleware` interrupts before the tool
  runs, so a naive setup asks the user to confirm a booking for 30 people in a 4-person room. The
  `when` predicate runs the same validation without writing and pauses only valid requests.
- **The predicate runs again on resume.** LangGraph replays the node when the user answers, so the
  check runs twice. That turned out to be a feature: if the slot was taken while the card was
  open, the second check fails, the tool refuses, and nothing is written.
- **Time zones in SQLite.** SQLite drops time zone information, which breaks comparisons between
  stored and requested times. A small `UtcDateTime` type stores naive UTC and returns aware UTC.
- **"Free" at night.** A multi-day schedule reported the night as free. Schedules are now clipped
  to business hours day by day, so "free" always means "bookable".
- **Testing an agent without a model.** LangChain's fake chat model cannot bind tools; a
  three-line subclass scripts exact tool calls, so the confirmation flow is tested
  deterministically in milliseconds.
- **A public demo spends real money.** A per-user message limit, a limit of six model calls per
  message and a budgeted OpenAI project keep the cost bounded.
- **A confirmation gate that failed open.** Reviewing the agent showed that a model sending
  `attendees` as the text `"4"` made the dry run raise a type error, which the gate read as
  "invalid, do not pause"; the tool's schema then converted `"4"` to `4` and wrote the booking
  without confirmation. The gate now validates the raw arguments with the tool's own schema
  first, so it checks exactly what the tool will execute, and a regression test pins it.
- **An assistant that answered in the office's language.** Running each eval ten times showed
  that English requests which could not be booked got a Spanish answer 4 times in 10, and once a
  Portuguese one. The prompt placed the office in Cubo Itaú, Montevideo, and the model localized
  its answer to the office instead of to the user. Measuring prompt variants one at a time showed
  that moving the language rule first helped (9 of 20 to 3 of 20) and that removing the location,
  which the model never needed, finished the job (no Spanish or Portuguese answer in the final measurement; the one
  failure in ten there was an empty reply, a separate and rarer behavior). The process
  lesson: a cheaper stand-in model is useful to simulate conversations and catch regressions
  for free, but only the production model's own failures should drive changes to its prompt.
- **A model that filled in what the user never said.** One eval run in twenty booked a room for
  a request that never mentioned how many people attend. The system prompt already forbade
  guessing; repeating the rule in the tool's argument descriptions, where the model actually
  fills the value, fixed it (30 of 30 runs of that case in the final measurement). The confirmation card stays as the
  safety net: the user sees the number before anything is written.

## Assumptions

Room capacities A=4, B=6, C=8, D=12, E=20; bookable hours 08:00–20:00 on any day; a single time
zone, `America/Montevideo`. All three are configuration.

## Limitations and next steps

- Conversations live in memory: a redeploy forgets them (bookings persist). Next step: a Postgres
  checkpointer, which also allows more than one replica.
- Tables are created at startup; next step: Alembic migrations.
- No token streaming; the page shows a "Thinking…" state instead.
- Recurring bookings and per-user time zones are out of scope.
- The walkthrough notebook was executed before the final prompt revision; its eval section shows
  the 11 evals that existed then (there are 16 now, measured in the commits that changed the prompt and allowed several bookings in one step).
