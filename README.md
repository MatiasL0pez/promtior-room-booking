# Room Booking Assistant

[![CI](https://github.com/MatiasL0pez/promtior-room-booking/actions/workflows/ci.yml/badge.svg)](https://github.com/MatiasL0pez/promtior-room-booking/actions/workflows/ci.yml)

A chatbot that books the meeting rooms of the Cubo Itaú office through tool calling, built for the
Promtior AI Engineer technical challenge.

**Live demo:** https://promtior-room-booking-production.up.railway.app (sign in as `User1` or
`User2`, password `TechnicalChallengePromtior`). API documentation at
[`/docs`](https://promtior-room-booking-production.up.railway.app/docs).

## What it does

- Checks which rooms are free for a time range, shows a room's schedule, and lists your upcoming
  bookings under the reply.
- Books a room (title and attendees required) and cancels your own bookings, always after you
  confirm on a card. Several bookings in one message ("room D from 11 to 13 for the next 7 days")
  arrive on one card with a single Confirm.
- Enforces every rule in code and in the database: five rooms A–E with their capacity, 30-minute
  slots, at most 3 hours, no overlaps, business hours.

## How it works

FastAPI → LangChain `create_agent` (LangGraph runtime, OpenAI `gpt-6-luna`) → five tools →
`BookingService` → SQLite or Postgres. The user reaches the tools through runtime context, never
as an argument the model could forge. Writes pause for confirmation only when a dry run says
they will succeed. Details in [doc/architecture.md](doc/architecture.md), the reasoning in
[doc/README.md](doc/README.md), and every technology with runnable code in
[doc/walkthrough.ipynb](doc/walkthrough.ipynb).

## Run it locally

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env        # set JWT_SECRET and OPENAI_API_KEY
uv sync
uv run uvicorn app.main:create_app --factory --reload
```
Open http://localhost:8000.

## Tests and evals

```bash
uv run pytest                          # unit tests: rules, API, agent wiring with a scripted model (no key needed)
uv run pytest -m eval                  # evals with the real model (needs OPENAI_API_KEY)
EVAL_REPEAT=10 uv run pytest -m eval   # every eval 10 times, with a pass rate per case
```

Last measurement (gpt-6-luna, 10 runs per case, commit `5b77ad6`): 14 of 17 cases at 10/10. Three
cases at 9/10: seven bookings confirmed on one card, the alternative offered in words, and the reply
language after a failed booking. A later fix made the reply after a confirmation follow the user's
language: 19 of 19 English replies, up from 12 of 15.

## Deploy

Railway: deploy this repository (it builds the `Dockerfile`), add a PostgreSQL database, and set
`DATABASE_URL=${{Postgres.DATABASE_URL}}`, `OPENAI_API_KEY`, `JWT_SECRET`.

## Repository map

| Path | What is there |
|---|---|
| `app/booking.py` | every booking rule |
| `app/agent.py` | tools, system prompt, confirmation gate |
| `app/chat.py` | conversations, decisions, rate limit |
| `app/main.py` | FastAPI routes |
| `app/static/index.html` | the web page |
| `tests/`, `evals/` | unit tests and evals |
| `doc/` | overview, architecture, notebook, design spec and plan |
