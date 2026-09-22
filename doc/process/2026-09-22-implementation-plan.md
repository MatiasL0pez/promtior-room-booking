# Room Booking Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deployed chatbot that books Cubo Itaú meeting rooms through tool calling, with every booking rule enforced by code and the database.

**Architecture:** FastAPI serves a login, a read-only REST API, a chat API and a static web page. A LangChain `create_agent` (LangGraph runtime, OpenAI model) calls five tools that delegate to `BookingService`, which owns every rule; a slot-level primary key makes double booking impossible. The logged-in user reaches the tools through runtime context, and `HumanInTheLoopMiddleware` pauses valid writes for confirmation.

**Tech Stack:** Python 3.13, uv, FastAPI, SQLAlchemy 2 (SQLite locally, Postgres on Railway), pwdlib[argon2], PyJWT, LangChain 1.4 / LangGraph 1.2, langchain-openai, pytest, ruff, Docker, Railway, GitHub Actions.

**Spec:** `doc/process/2026-09-22-design-spec.md`

## Global Constraints

- Python `>=3.13`; dependencies managed with `uv` and pinned by `uv.lock`.
- Everything in English: code, names, docs, notebook, commit messages.
- Code style: descriptive names, no leading underscores, code that reads top to bottom, a helper only when two places use it, explicit `if` blocks instead of chained ternaries, no comments except one line when the reason is not obvious.
- Every booking rule lives in `BookingService`. Tools and routes never validate booking rules; the prompt never enforces them.
- The user comes only from the JWT → `CurrentUser` → `ToolRuntime` context. It is never a tool argument.
- Tools and the REST API speak local ISO 8601 `YYYY-MM-DDTHH:MM` in `America/Montevideo`; the database stores naive UTC through `UtcDateTime`.
- Error payload shape: `{"ok": false, "error": "<CODE>", "message": "<text>", ...details}`.
- Capacities: A=4, B=6, C=8, D=12, E=20. Bookable hours 08:00–20:00. Slots of 30 minutes, max 3 hours.
- One commit per task. The message says what was missing or failing and why. No `Co-Authored-By` and no AI signature.
- Library versions verified on 2026-09-22: langchain 1.4.2, langgraph 1.2.12, langchain-openai 1.6.3, fastapi 0.141.1, SQLAlchemy 2.0.54, pwdlib 0.3.1, PyJWT 2.14.0, pydantic-settings 2.15.0, uv 0.12.17, actions/checkout v7, astral-sh/setup-uv v10.
- Default model `gpt-6-luna` through the Responses API, reasoning effort `low`.
- Before every lint check run `uv run ruff check --fix . && uv run ruff format .`: steps that say "add to the imports" rely on it to sort imports and wrap lines.
- Execution rhythm: one task at a time; show the diff with `git dp`; wait for the author's OK; then commit.

## Review Focus

1. The model sends datetimes with seconds or an offset (`2026-09-23T10:00:00-03:00`, `...T13:00:00+00:00`): they must mean the same instant as `2026-09-23T10:00`. Pinned in Task 3.
2. Bookings that touch the edges of the day (start 08:00, end 20:00) are accepted, and 19:30–20:30 is rejected. Pinned in Task 3.
3. A schedule whose first booking starts at opening time, or whose range spans several days, must not contain zero-length ranges or "free" nights. Pinned in Task 5.
4. The model sends a lowercase room (`"b"`): the booking and the confirmation card must say room B. Pinned in Task 8.
5. A double click on Confirm must create exactly one booking and answer the second click with 409. Pinned in Task 9.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.python-version` (by uv), `uv.lock` (by uv), `.gitattributes`, `.env.example`, `app/__init__.py`, `app/config.py`, `app/main.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_config.py`, `tests/test_api.py`

**Interfaces:**
- Produces: `Settings` (fields `jwt_secret`, `openai_api_key`, `openai_model`, `openai_reasoning_effort`, `database_url`, `seed_password`, `office_timezone`, `opening_hour`, `closing_hour`, `token_lifetime_minutes`, `messages_per_window`, `message_window_minutes`; property `sqlalchemy_database_url`); `create_app(settings: Settings | None = None) -> FastAPI`; fixtures `settings`, `client`.

- [ ] **Step 1: Install uv**

Run (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
Then in Git Bash: `export PATH="$HOME/.local/bin:$PATH" && uv --version`
Expected: `uv 0.12.x`

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "room-booking-assistant"
version = "1.0.0"
description = "Conversational meeting room booking assistant with tool calling"
requires-python = ">=3.13"
dependencies = []

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-m 'not eval'"
markers = ["eval: talks to the real OpenAI model; needs OPENAI_API_KEY"]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
ignore = ["E501"]
```

- [ ] **Step 3: Add the dependencies**

```bash
uv python pin 3.13
uv add fastapi "uvicorn[standard]" sqlalchemy "psycopg[binary]" pydantic-settings pyjwt "pwdlib[argon2]" python-multipart tzdata langchain langchain-openai
uv add --dev pytest httpx ruff
```
Expected: `pyproject.toml` lists them, `uv.lock` and `.python-version` exist.

- [ ] **Step 4: Write `.gitattributes`, `.env.example` and the empty packages**

`.gitattributes`:
```
* text=auto eol=lf
*.png binary
```

`.env.example`:
```
JWT_SECRET=replace-with-a-long-random-string
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-6-luna
OPENAI_REASONING_EFFORT=low
# DATABASE_URL=postgresql://user:password@host:5432/database
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=lsv2-your-key
```

`app/__init__.py` and `tests/__init__.py`: empty files.

- [ ] **Step 5: Write the failing tests**

`tests/conftest.py`:
```python
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        jwt_secret="test-secret",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
    )


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings))
```

`tests/test_config.py`:
```python
import pytest

from app.config import Settings


@pytest.mark.parametrize(
    ("database_url", "expected"),
    [
        ("postgresql://user:pw@host:5432/db", "postgresql+psycopg://user:pw@host:5432/db"),
        ("postgres://user:pw@host:5432/db", "postgresql+psycopg://user:pw@host:5432/db"),
        ("sqlite:///./room_booking.db", "sqlite:///./room_booking.db"),
    ],
)
def test_database_url_gets_the_psycopg_driver(database_url, expected):
    settings = Settings(_env_file=None, jwt_secret="x", database_url=database_url)
    assert settings.sqlalchemy_database_url == expected


def test_jwt_secret_is_required(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(ValueError):
        Settings(_env_file=None)
```

`tests/test_api.py`:
```python
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `uv run pytest -q`
Expected: collection error `ModuleNotFoundError: No module named 'app.config'`.

- [ ] **Step 7: Write `app/config.py` and `app/main.py`**

`app/config.py`:
```python
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    jwt_secret: str
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-6-luna"
    openai_reasoning_effort: str = "low"
    database_url: str = "sqlite:///./room_booking.db"
    seed_password: str = "TechnicalChallengePromtior"
    office_timezone: str = "America/Montevideo"
    opening_hour: int = 8
    closing_hour: int = 20
    token_lifetime_minutes: int = 480
    messages_per_window: int = 30
    message_window_minutes: int = 10

    @property
    def sqlalchemy_database_url(self) -> str:
        for prefix in ("postgres://", "postgresql://"):
            if self.database_url.startswith(prefix):
                return "postgresql+psycopg://" + self.database_url.removeprefix(prefix)
        return self.database_url
```

`app/main.py`:
```python
from fastapi import FastAPI

from app.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    app = FastAPI(title="Room Booking Assistant")
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
```

- [ ] **Step 8: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `5 passed`, `All checks passed!`, `... files already formatted` (run `uv run ruff format .` first if it reports files).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitattributes .env.example app tests
git commit -m "Add project scaffold

There was nothing to run or test. Settings fail fast without JWT_SECRET and normalize
Railway's postgresql:// URL to the psycopg driver; LF line endings keep the Linux image
and the Windows checkout identical."
```

---

### Task 2: Tables and seed

**Files:**
- Create: `app/models.py`, `app/db.py`, `tests/test_db.py`

**Interfaces:**
- Consumes: `Settings.sqlalchemy_database_url`, `Settings.seed_password`.
- Produces: models `User`, `Room`, `Booking`, `BookingSlot`, type `UtcDateTime`; `ROOM_CAPACITIES: dict[str, int]`; `password_hasher: PasswordHash`; `create_session_factory(database_url: str) -> sessionmaker[Session]`; `seed(session_factory, password: str) -> None`.

- [ ] **Step 1: Write the failing test**

`tests/test_db.py`:
```python
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db import ROOM_CAPACITIES, create_session_factory, password_hasher, seed
from app.models import Booking, BookingSlot, Room, User

MONTEVIDEO = ZoneInfo("America/Montevideo")


@pytest.fixture
def session_factory(settings):
    session_factory = create_session_factory(settings.sqlalchemy_database_url)
    seed(session_factory, settings.seed_password)
    return session_factory


def add_booking(session, user_id: int, start: datetime) -> Booking:
    booking = Booking(
        room_id="A",
        user_id=user_id,
        title="Test",
        attendees=2,
        start_at=start,
        end_at=start + timedelta(minutes=30),
        created_at=start,
    )
    session.add(booking)
    session.flush()
    return booking


def test_seed_creates_rooms_and_users_once(settings, session_factory):
    seed(session_factory, settings.seed_password)
    with session_factory() as session:
        rooms = {room.id: room.capacity for room in session.scalars(select(Room))}
        users = list(session.scalars(select(User).order_by(User.id)))
    assert rooms == ROOM_CAPACITIES
    assert [user.username for user in users] == ["User1", "User2"]
    assert password_hasher.verify("TechnicalChallengePromtior", users[0].password_hash)


def test_a_slot_is_held_by_one_booking_only(session_factory):
    slot_start = datetime(2026, 9, 23, 10, 0, tzinfo=MONTEVIDEO)
    with session_factory.begin() as session:
        first = add_booking(session, 1, slot_start)
        session.add(BookingSlot(room_id="A", slot_start=slot_start, booking_id=first.id))
    with pytest.raises(IntegrityError), session_factory.begin() as session:
        second = add_booking(session, 2, slot_start)
        session.add(
            BookingSlot(room_id="A", slot_start=slot_start.astimezone(UTC), booking_id=second.id)
        )


def test_datetimes_are_stored_as_the_same_instant(session_factory):
    start = datetime(2026, 9, 23, 10, 0, tzinfo=MONTEVIDEO)
    with session_factory.begin() as session:
        booking_id = add_booking(session, 1, start).id
    with session_factory() as session:
        stored = session.get(Booking, booking_id)
    assert stored.start_at == start
    assert stored.start_at.tzinfo is UTC
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_db.py -q`
Expected: `ModuleNotFoundError: No module named 'app.db'`.

- [ ] **Step 3: Write `app/models.py` and `app/db.py`**

`app/models.py`:
```python
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


# SQLite drops time zones, so every datetime is stored as naive UTC and read back as aware UTC.
class UtcDateTime(TypeDecorator):
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))


class Room(Base):
    __tablename__ = "rooms"

    id: Mapped[str] = mapped_column(String(1), primary_key=True)
    capacity: Mapped[int]


class Booking(Base):
    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[str] = mapped_column(ForeignKey("rooms.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(120))
    attendees: Mapped[int]
    start_at: Mapped[datetime] = mapped_column(UtcDateTime)
    end_at: Mapped[datetime] = mapped_column(UtcDateTime)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)
    cancelled_at: Mapped[datetime | None] = mapped_column(UtcDateTime)


class BookingSlot(Base):
    __tablename__ = "booking_slots"

    room_id: Mapped[str] = mapped_column(ForeignKey("rooms.id"), primary_key=True)
    slot_start: Mapped[datetime] = mapped_column(UtcDateTime, primary_key=True)
    booking_id: Mapped[int] = mapped_column(ForeignKey("bookings.id"))
```

`app/db.py`:
```python
from pwdlib import PasswordHash
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base, Room, User

ROOM_CAPACITIES = {"A": 4, "B": 6, "C": 8, "D": 12, "E": 20}
USERNAMES = ["User1", "User2"]

password_hasher = PasswordHash.recommended()


def create_session_factory(database_url: str) -> sessionmaker[Session]:
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def seed(session_factory: sessionmaker[Session], password: str) -> None:
    with session_factory.begin() as session:
        for room_id, capacity in ROOM_CAPACITIES.items():
            if session.get(Room, room_id) is None:
                session.add(Room(id=room_id, capacity=capacity))
        for username in USERNAMES:
            if session.scalar(select(User).where(User.username == username)) is None:
                session.add(User(username=username, password_hash=password_hasher.hash(password)))
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `8 passed`, lint clean.

- [ ] **Step 5: Commit**

```bash
git add app/models.py app/db.py tests/test_db.py
git commit -m "Add tables, slot primary key and idempotent seed

The rule 'a slot can only be held by one booking' needs a guarantee that survives two
requests at the same time, so every 30-minute slot is a row keyed by (room_id, slot_start).
SQLite loses time zones, so datetimes go through UtcDateTime to compare as instants."
```

---

### Task 3: Booking rules for creating

**Files:**
- Create: `app/booking.py`, `tests/support.py`, `tests/test_booking.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: `create_session_factory`, `seed`, models from Task 2; `Settings.office_timezone`, `opening_hour`, `closing_hour`.
- Produces: `BookingError(code, message, **details)` with `.code`, `.message`, `.details`, `.as_dict()`; `BookingRequest(room_id, start, end, title, attendees)`; `active_bookings_overlapping(start, end) -> list`; `BookingService(session_factory, settings, clock)` with attributes `session_factory`, `clock`, `timezone`, `opening_hour`, `closing_hour` and methods `parse_local_datetime(text) -> datetime`, `to_local_text(value) -> str`, `describe(booking) -> dict`, `check_business_hours(start, end) -> None`, `check_create(request) -> None`, `create(user_id, request) -> dict`. Test support: `MONTEVIDEO`, `FIXED_NOW`, `USER1_ID`, `USER2_ID`, `FakeClock`, `local(text)`, `booking_request(**changes)`; fixtures `clock`, `service`.

- [ ] **Step 1: Write the test support and fixtures**

`tests/support.py`:
```python
from datetime import datetime
from zoneinfo import ZoneInfo

from app.booking import BookingRequest

MONTEVIDEO = ZoneInfo("America/Montevideo")
FIXED_NOW = datetime(2026, 9, 22, 9, 0, tzinfo=MONTEVIDEO)
USER1_ID = 1
USER2_ID = 2


class FakeClock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def local(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=MONTEVIDEO)


def booking_request(
    room_id: str = "B",
    start: str = "2026-09-23T10:00",
    end: str = "2026-09-23T11:30",
    title: str = "Interview with John Doe",
    attendees: int = 4,
) -> BookingRequest:
    return BookingRequest(
        room_id=room_id, start=local(start), end=local(end), title=title, attendees=attendees
    )
```

`tests/conftest.py` (whole file):
```python
import pytest
from fastapi.testclient import TestClient

from app.booking import BookingService
from app.config import Settings
from app.db import create_session_factory, seed
from app.main import create_app
from tests.support import FIXED_NOW, FakeClock


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        jwt_secret="test-secret",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(FIXED_NOW)


@pytest.fixture
def service(settings, clock) -> BookingService:
    session_factory = create_session_factory(settings.sqlalchemy_database_url)
    seed(session_factory, settings.seed_password)
    return BookingService(session_factory, settings, clock)


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings))
```

- [ ] **Step 2: Write the failing tests**

`tests/test_booking.py`:
```python
import pytest
from sqlalchemy import select

from app.booking import BookingError
from app.models import Booking
from tests.support import USER1_ID, USER2_ID, booking_request, local


def test_valid_booking_is_created_in_local_time(service):
    booking = service.create(USER1_ID, booking_request())
    assert booking == {
        "booking_id": 1,
        "room": "B",
        "start": "2026-09-23T10:00",
        "end": "2026-09-23T11:30",
        "title": "Interview with John Doe",
        "attendees": 4,
    }


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"room_id": "Z"}, "ROOM_NOT_FOUND"),
        ({"end": "2026-09-23T10:00"}, "INVALID_TIME_RANGE"),
        ({"start": "2026-09-23T10:15"}, "NOT_ALIGNED"),
        ({"end": "2026-09-23T13:30"}, "TOO_LONG"),
        ({"start": "2026-09-21T10:00", "end": "2026-09-21T11:00"}, "IN_THE_PAST"),
        ({"start": "2026-09-23T07:30", "end": "2026-09-23T08:30"}, "OUTSIDE_BUSINESS_HOURS"),
        ({"start": "2026-09-23T19:30", "end": "2026-09-23T20:30"}, "OUTSIDE_BUSINESS_HOURS"),
        ({"attendees": 0}, "INVALID_ATTENDEES"),
        ({"attendees": 7}, "CAPACITY_EXCEEDED"),
        ({"title": "   "}, "TITLE_REQUIRED"),
        ({"title": "x" * 121}, "TITLE_TOO_LONG"),
    ],
)
def test_invalid_booking_is_rejected_with_its_code(service, changes, code):
    with pytest.raises(BookingError) as error:
        service.create(USER1_ID, booking_request(**changes))
    assert error.value.code == code


def test_three_hours_and_the_edges_of_the_day_are_bookable(service):
    service.create(USER1_ID, booking_request(start="2026-09-23T08:00", end="2026-09-23T11:00"))
    service.create(USER1_ID, booking_request(start="2026-09-23T17:00", end="2026-09-23T20:00"))


def test_a_booking_can_start_when_the_previous_one_ends(service):
    service.create(USER1_ID, booking_request(start="2026-09-23T10:00", end="2026-09-23T11:30"))
    booking = service.create(
        USER2_ID, booking_request(start="2026-09-23T11:30", end="2026-09-23T12:00")
    )
    assert booking["start"] == "2026-09-23T11:30"


def test_overlap_is_rejected_with_the_conflicting_range(service):
    service.create(USER1_ID, booking_request(start="2026-09-23T10:00", end="2026-09-23T11:30"))
    with pytest.raises(BookingError) as error:
        service.create(USER2_ID, booking_request(start="2026-09-23T11:00", end="2026-09-23T12:00"))
    assert error.value.code == "SLOT_TAKEN"
    assert error.value.details == {"conflict": {"start": "2026-09-23T10:00", "end": "2026-09-23T11:30"}}


def test_capacity_error_carries_the_capacity(service):
    with pytest.raises(BookingError) as error:
        service.create(USER1_ID, booking_request(attendees=7))
    assert error.value.as_dict() == {
        "ok": False,
        "error": "CAPACITY_EXCEEDED",
        "message": "Room B holds at most 6 people.",
        "capacity": 6,
    }


def test_check_create_validates_without_writing(service):
    service.check_create(booking_request())
    with service.session_factory() as session:
        assert session.scalars(select(Booking)).all() == []


def test_slot_taken_when_a_concurrent_booking_wins_the_race(service, monkeypatch):
    service.create(USER1_ID, booking_request())
    monkeypatch.setattr(service, "check_create", lambda request: None)
    with pytest.raises(BookingError) as error:
        service.create(USER2_ID, booking_request())
    assert error.value.code == "SLOT_TAKEN"


@pytest.mark.parametrize(
    "text",
    [
        "2026-09-23T10:00",
        "2026-09-23T10:00:00",
        "2026-09-23T10:00:00-03:00",
        "2026-09-23T13:00:00+00:00",
    ],
)
def test_local_datetimes_parse_to_the_same_instant(service, text):
    assert service.parse_local_datetime(text) == local("2026-09-23T10:00")


def test_unreadable_datetime_is_a_booking_error(service):
    with pytest.raises(BookingError) as error:
        service.parse_local_datetime("tomorrow at ten")
    assert error.value.code == "INVALID_DATETIME"
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_booking.py -q`
Expected: `ModuleNotFoundError: No module named 'app.booking'`.

- [ ] **Step 4: Write `app/booking.py`**

```python
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models import Booking, BookingSlot, Room

SLOT_LENGTH = timedelta(minutes=30)
MAX_DURATION = timedelta(hours=3)
MAX_TITLE_LENGTH = 120


class BookingError(Exception):
    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict:
        return {"ok": False, "error": self.code, "message": self.message, **self.details}


@dataclass(frozen=True)
class BookingRequest:
    room_id: str
    start: datetime
    end: datetime
    title: str
    attendees: int


def active_bookings_overlapping(start: datetime, end: datetime) -> list:
    return [Booking.cancelled_at.is_(None), Booking.start_at < end, Booking.end_at > start]


class BookingService:
    def __init__(self, session_factory: sessionmaker[Session], settings: Settings, clock):
        self.session_factory = session_factory
        self.clock = clock
        self.timezone = ZoneInfo(settings.office_timezone)
        self.opening_hour = settings.opening_hour
        self.closing_hour = settings.closing_hour

    def parse_local_datetime(self, text: str) -> datetime:
        try:
            value = datetime.fromisoformat(text)
        except (TypeError, ValueError):
            raise BookingError(
                "INVALID_DATETIME",
                f"'{text}' is not an ISO 8601 date and time like 2026-09-23T10:00.",
            ) from None
        if value.tzinfo is None:
            return value.replace(tzinfo=self.timezone)
        return value.astimezone(self.timezone)

    def to_local_text(self, value: datetime) -> str:
        return value.astimezone(self.timezone).strftime("%Y-%m-%dT%H:%M")

    def describe(self, booking: Booking) -> dict:
        return {
            "booking_id": booking.id,
            "room": booking.room_id,
            "start": self.to_local_text(booking.start_at),
            "end": self.to_local_text(booking.end_at),
            "title": booking.title,
            "attendees": booking.attendees,
        }

    def check_business_hours(self, start: datetime, end: datetime) -> None:
        day = start.astimezone(self.timezone).date()
        opening = datetime.combine(day, time(self.opening_hour), self.timezone)
        closing = datetime.combine(day, time(self.closing_hour), self.timezone)
        if start < opening or end > closing:
            raise BookingError(
                "OUTSIDE_BUSINESS_HOURS",
                f"Rooms can be booked from {self.opening_hour:02d}:00 to "
                f"{self.closing_hour:02d}:00 within a single day.",
                opening=f"{self.opening_hour:02d}:00",
                closing=f"{self.closing_hour:02d}:00",
            )

    def check_create(self, request: BookingRequest) -> None:
        with self.session_factory() as session:
            room = session.get(Room, request.room_id)
            if room is None:
                valid_rooms = list(session.scalars(select(Room.id).order_by(Room.id)))
                raise BookingError(
                    "ROOM_NOT_FOUND",
                    f"Room {request.room_id} does not exist.",
                    valid_rooms=valid_rooms,
                )
            if request.end <= request.start:
                raise BookingError("INVALID_TIME_RANGE", "The end must be after the start.")
            for value in (request.start, request.end):
                local_value = value.astimezone(self.timezone)
                if local_value.minute not in (0, 30) or local_value.second or local_value.microsecond:
                    raise BookingError(
                        "NOT_ALIGNED", "Bookings start and end on the hour or the half hour."
                    )
            if request.end - request.start > MAX_DURATION:
                raise BookingError("TOO_LONG", "A booking lasts at most 3 hours.", max_hours=3)
            now = self.clock()
            if request.start < now:
                raise BookingError(
                    "IN_THE_PAST",
                    "The booking would start in the past.",
                    now=self.to_local_text(now),
                )
            self.check_business_hours(request.start, request.end)
            if request.attendees < 1:
                raise BookingError("INVALID_ATTENDEES", "A booking needs at least one attendee.")
            if request.attendees > room.capacity:
                raise BookingError(
                    "CAPACITY_EXCEEDED",
                    f"Room {room.id} holds at most {room.capacity} people.",
                    capacity=room.capacity,
                )
            title = request.title.strip()
            if not title:
                raise BookingError("TITLE_REQUIRED", "Every booking needs a title.")
            if len(title) > MAX_TITLE_LENGTH:
                raise BookingError(
                    "TITLE_TOO_LONG",
                    f"A title has at most {MAX_TITLE_LENGTH} characters.",
                    max_length=MAX_TITLE_LENGTH,
                )
            conflict = session.scalar(
                select(Booking)
                .where(
                    Booking.room_id == room.id,
                    *active_bookings_overlapping(request.start, request.end),
                )
                .order_by(Booking.start_at)
            )
            if conflict is not None:
                conflict_start = self.to_local_text(conflict.start_at)
                conflict_end = self.to_local_text(conflict.end_at)
                raise BookingError(
                    "SLOT_TAKEN",
                    f"Room {room.id} is already booked from {conflict_start[11:]} "
                    f"to {conflict_end[11:]}.",
                    conflict={"start": conflict_start, "end": conflict_end},
                )

    def create(self, user_id: int, request: BookingRequest) -> dict:
        self.check_create(request)
        booking = Booking(
            room_id=request.room_id,
            user_id=user_id,
            title=request.title.strip(),
            attendees=request.attendees,
            start_at=request.start,
            end_at=request.end,
            created_at=self.clock(),
        )
        try:
            with self.session_factory.begin() as session:
                session.add(booking)
                session.flush()
                slot_start = request.start
                while slot_start < request.end:
                    session.add(
                        BookingSlot(
                            room_id=request.room_id, slot_start=slot_start, booking_id=booking.id
                        )
                    )
                    slot_start += SLOT_LENGTH
        except IntegrityError:
            raise BookingError(
                "SLOT_TAKEN",
                f"Room {request.room_id} was just booked by someone else for that time.",
            ) from None
        return self.describe(booking)
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `31 passed`, lint clean.

- [ ] **Step 6: Commit**

```bash
git add app/booking.py tests/support.py tests/conftest.py tests/test_booking.py
git commit -m "Add booking rules for creating a booking

Nothing enforced the challenge rules. BookingService validates room, alignment, duration,
past, business hours, capacity, title and overlap, each with a code and the data the
assistant needs to explain it; check_create runs the same checks without writing so the
confirmation step can ask only about bookings that will succeed."
```

---

### Task 4: Cancelling

**Files:**
- Modify: `app/booking.py`, `tests/test_booking.py`

**Interfaces:**
- Consumes: `BookingService` from Task 3.
- Produces: `BookingService.check_cancel(user_id, booking_id) -> dict` (the booking as `describe` returns it) and `BookingService.cancel(user_id, booking_id) -> dict`.

- [ ] **Step 1: Append the failing tests to `tests/test_booking.py`**

Add `BookingSlot` to the models import (`from app.models import Booking, BookingSlot`) and append:
```python
def test_owner_cancels_and_the_slots_become_free(service):
    booking = service.create(USER1_ID, booking_request())
    assert service.cancel(USER1_ID, booking["booking_id"]) == booking
    again = service.create(USER2_ID, booking_request())
    assert again["start"] == "2026-09-23T10:00"


def test_a_cancelled_booking_stays_in_history_without_slots(service):
    booking = service.create(USER1_ID, booking_request())
    service.cancel(USER1_ID, booking["booking_id"])
    with service.session_factory() as session:
        stored = session.get(Booking, booking["booking_id"])
        slots = session.scalars(
            select(BookingSlot).where(BookingSlot.booking_id == booking["booking_id"])
        ).all()
    assert stored.cancelled_at is not None
    assert slots == []


def test_only_the_owner_can_cancel(service):
    booking = service.create(USER1_ID, booking_request())
    with pytest.raises(BookingError) as error:
        service.cancel(USER2_ID, booking["booking_id"])
    assert error.value.code == "NOT_OWNER"


def test_an_unknown_booking_cannot_be_cancelled(service):
    with pytest.raises(BookingError) as error:
        service.cancel(USER1_ID, 999)
    assert error.value.code == "BOOKING_NOT_FOUND"


def test_a_booking_is_cancelled_once(service):
    booking = service.create(USER1_ID, booking_request())
    service.cancel(USER1_ID, booking["booking_id"])
    with pytest.raises(BookingError) as error:
        service.cancel(USER1_ID, booking["booking_id"])
    assert error.value.code == "ALREADY_CANCELLED"


def test_a_started_booking_cannot_be_cancelled(service, clock):
    booking = service.create(USER1_ID, booking_request())
    clock.now = local("2026-09-23T10:00")
    with pytest.raises(BookingError) as error:
        service.cancel(USER1_ID, booking["booking_id"])
    assert error.value.code == "ALREADY_STARTED"


def test_check_cancel_describes_without_writing(service):
    booking = service.create(USER1_ID, booking_request())
    assert service.check_cancel(USER1_ID, booking["booking_id"]) == booking
    with service.session_factory() as session:
        assert session.get(Booking, booking["booking_id"]).cancelled_at is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_booking.py -q`
Expected: 7 failures with `AttributeError: 'BookingService' object has no attribute 'cancel'` (or `check_cancel`).

- [ ] **Step 3: Implement**

In `app/booking.py` change the sqlalchemy import to `from sqlalchemy import delete, select, update` and add these methods at the end of `BookingService`:
```python
    def check_cancel(self, user_id: int, booking_id: int) -> dict:
        with self.session_factory() as session:
            booking = session.get(Booking, booking_id)
            if booking is None:
                raise BookingError("BOOKING_NOT_FOUND", f"There is no booking {booking_id}.")
            if booking.user_id != user_id:
                raise BookingError(
                    "NOT_OWNER", "Only the person who made a booking can cancel it."
                )
            if booking.cancelled_at is not None:
                raise BookingError(
                    "ALREADY_CANCELLED", f"Booking {booking_id} is already cancelled."
                )
            if booking.start_at <= self.clock():
                raise BookingError("ALREADY_STARTED", f"Booking {booking_id} has already started.")
            return self.describe(booking)

    def cancel(self, user_id: int, booking_id: int) -> dict:
        cancelled = self.check_cancel(user_id, booking_id)
        with self.session_factory.begin() as session:
            result = session.execute(
                update(Booking)
                .where(Booking.id == booking_id, Booking.cancelled_at.is_(None))
                .values(cancelled_at=self.clock())
            )
            if result.rowcount == 0:
                raise BookingError(
                    "ALREADY_CANCELLED", f"Booking {booking_id} is already cancelled."
                )
            session.execute(delete(BookingSlot).where(BookingSlot.booking_id == booking_id))
        return cancelled
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `38 passed`, lint clean.

- [ ] **Step 5: Commit**

```bash
git add app/booking.py tests/test_booking.py
git commit -m "Add cancelling own bookings

Users could not cancel. Only the owner cancels an active booking that has not started; the
booking keeps its history with cancelled_at while its slot rows are deleted in the same
transaction, and the conditional update makes two simultaneous cancels count once."
```

---

### Task 5: Queries — rooms, availability, schedule, my bookings

**Files:**
- Modify: `app/booking.py`, `tests/test_booking.py`

**Interfaces:**
- Consumes: `BookingService` from Tasks 3–4.
- Produces: `BookingService.rooms() -> list[dict]` (`{"room", "capacity"}`), `available_rooms(start, end, attendees: int | None = None) -> list[dict]`, `room_schedule(viewer_user_id, room_id, start, end) -> dict` (`{"room", "capacity", "ranges": [...]}`; a range is `{"start", "end", "status": "free"}` or `{"start", "end", "status": "occupied", "mine": bool}` plus `"booking_id"` and `"title"` when `mine`), `bookings_of(user_id) -> list[dict]`.

- [ ] **Step 1: Append the failing tests to `tests/test_booking.py`**

```python
def test_rooms_lists_the_capacities(service):
    assert service.rooms() == [
        {"room": "A", "capacity": 4},
        {"room": "B", "capacity": 6},
        {"room": "C", "capacity": 8},
        {"room": "D", "capacity": 12},
        {"room": "E", "capacity": 20},
    ]


def test_available_rooms_skip_busy_and_small_rooms(service):
    service.create(USER1_ID, booking_request(room_id="C", attendees=5))
    start, end = local("2026-09-23T10:00"), local("2026-09-23T11:00")
    assert [room["room"] for room in service.available_rooms(start, end)] == ["A", "B", "D", "E"]
    assert [room["room"] for room in service.available_rooms(start, end, attendees=8)] == ["D", "E"]


def test_available_rooms_outside_business_hours_is_an_error(service):
    with pytest.raises(BookingError) as error:
        service.available_rooms(local("2026-09-23T20:00"), local("2026-09-23T21:00"))
    assert error.value.code == "OUTSIDE_BUSINESS_HOURS"


def test_schedule_hides_other_peoples_titles(service):
    service.create(USER1_ID, booking_request(title="Interview"))
    service.create(
        USER2_ID,
        booking_request(start="2026-09-23T14:00", end="2026-09-23T15:00", title="Secret merger talks"),
    )
    schedule = service.room_schedule(
        USER1_ID, "B", local("2026-09-23T00:00"), local("2026-09-24T00:00")
    )
    assert schedule == {
        "room": "B",
        "capacity": 6,
        "ranges": [
            {"start": "2026-09-23T08:00", "end": "2026-09-23T10:00", "status": "free"},
            {
                "start": "2026-09-23T10:00",
                "end": "2026-09-23T11:30",
                "status": "occupied",
                "mine": True,
                "booking_id": 1,
                "title": "Interview",
            },
            {"start": "2026-09-23T11:30", "end": "2026-09-23T14:00", "status": "free"},
            {"start": "2026-09-23T14:00", "end": "2026-09-23T15:00", "status": "occupied", "mine": False},
            {"start": "2026-09-23T15:00", "end": "2026-09-23T20:00", "status": "free"},
        ],
    }
    assert "Secret" not in str(schedule)


def test_schedule_covers_business_hours_of_each_day_without_empty_ranges(service):
    service.create(USER1_ID, booking_request(start="2026-09-23T08:00", end="2026-09-23T09:00"))
    schedule = service.room_schedule(
        USER1_ID, "B", local("2026-09-23T00:00"), local("2026-09-24T23:00")
    )
    assert [(entry["start"], entry["end"], entry["status"]) for entry in schedule["ranges"]] == [
        ("2026-09-23T08:00", "2026-09-23T09:00", "occupied"),
        ("2026-09-23T09:00", "2026-09-23T20:00", "free"),
        ("2026-09-24T08:00", "2026-09-24T20:00", "free"),
    ]


@pytest.mark.parametrize(
    ("room_id", "end", "code"),
    [("Z", "2026-09-24T00:00", "ROOM_NOT_FOUND"), ("B", "2026-10-01T00:00", "RANGE_TOO_LARGE")],
)
def test_schedule_errors(service, room_id, end, code):
    with pytest.raises(BookingError) as error:
        service.room_schedule(USER1_ID, room_id, local("2026-09-23T00:00"), local(end))
    assert error.value.code == code


def test_bookings_of_lists_only_own_active_future_bookings(service, clock):
    service.create(USER1_ID, booking_request(start="2026-09-23T15:00", end="2026-09-23T16:00", title="Later"))
    service.create(USER1_ID, booking_request(title="Earlier"))
    dropped = service.create(USER1_ID, booking_request(room_id="C", title="Dropped", attendees=2))
    service.cancel(USER1_ID, dropped["booking_id"])
    service.create(USER2_ID, booking_request(room_id="D", title="Not mine"))
    assert [booking["title"] for booking in service.bookings_of(USER1_ID)] == ["Earlier", "Later"]
    clock.now = local("2026-09-23T12:00")
    assert [booking["title"] for booking in service.bookings_of(USER1_ID)] == ["Later"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_booking.py -q`
Expected: 8 failures with `AttributeError: 'BookingService' object has no attribute 'rooms'` (and the other query methods).

- [ ] **Step 3: Implement**

In `app/booking.py` add the constant below `MAX_DURATION`:
```python
MAX_QUERY_RANGE = timedelta(days=7)
```
and add these methods at the end of `BookingService`:
```python
    def rooms(self) -> list[dict]:
        with self.session_factory() as session:
            rooms = session.scalars(select(Room).order_by(Room.id))
            return [{"room": room.id, "capacity": room.capacity} for room in rooms]

    def available_rooms(
        self, start: datetime, end: datetime, attendees: int | None = None
    ) -> list[dict]:
        if end <= start:
            raise BookingError("INVALID_TIME_RANGE", "The end must be after the start.")
        self.check_business_hours(start, end)
        with self.session_factory() as session:
            busy_room_ids = set(
                session.scalars(
                    select(Booking.room_id).where(*active_bookings_overlapping(start, end))
                )
            )
            rooms = list(session.scalars(select(Room).order_by(Room.id)))
        available = []
        for room in rooms:
            if room.id in busy_room_ids:
                continue
            if attendees is not None and room.capacity < attendees:
                continue
            available.append({"room": room.id, "capacity": room.capacity})
        return available

    def room_schedule(
        self, viewer_user_id: int, room_id: str, start: datetime, end: datetime
    ) -> dict:
        if end <= start:
            raise BookingError("INVALID_TIME_RANGE", "The end must be after the start.")
        if end - start > MAX_QUERY_RANGE:
            raise BookingError("RANGE_TOO_LARGE", "A schedule covers at most 7 days.", max_days=7)
        with self.session_factory() as session:
            room = session.get(Room, room_id)
            if room is None:
                valid_rooms = list(session.scalars(select(Room.id).order_by(Room.id)))
                raise BookingError(
                    "ROOM_NOT_FOUND", f"Room {room_id} does not exist.", valid_rooms=valid_rooms
                )
            bookings = list(
                session.scalars(
                    select(Booking)
                    .where(Booking.room_id == room_id, *active_bookings_overlapping(start, end))
                    .order_by(Booking.start_at)
                )
            )
        ranges = []
        day = start.astimezone(self.timezone).date()
        last_day = end.astimezone(self.timezone).date()
        while day <= last_day:
            window_start = max(start, datetime.combine(day, time(self.opening_hour), self.timezone))
            window_end = min(end, datetime.combine(day, time(self.closing_hour), self.timezone))
            day += timedelta(days=1)
            if window_end <= window_start:
                continue
            cursor = window_start
            for booking in bookings:
                if booking.end_at <= cursor or booking.start_at >= window_end:
                    continue
                if booking.start_at > cursor:
                    ranges.append(
                        {
                            "start": self.to_local_text(cursor),
                            "end": self.to_local_text(booking.start_at),
                            "status": "free",
                        }
                    )
                occupied_end = min(booking.end_at, window_end)
                mine = booking.user_id == viewer_user_id
                occupied = {
                    "start": self.to_local_text(max(booking.start_at, cursor)),
                    "end": self.to_local_text(occupied_end),
                    "status": "occupied",
                    "mine": mine,
                }
                if mine:
                    occupied["booking_id"] = booking.id
                    occupied["title"] = booking.title
                ranges.append(occupied)
                cursor = occupied_end
            if cursor < window_end:
                ranges.append(
                    {
                        "start": self.to_local_text(cursor),
                        "end": self.to_local_text(window_end),
                        "status": "free",
                    }
                )
        return {"room": room.id, "capacity": room.capacity, "ranges": ranges}

    def bookings_of(self, user_id: int) -> list[dict]:
        with self.session_factory() as session:
            bookings = session.scalars(
                select(Booking)
                .where(
                    Booking.user_id == user_id,
                    Booking.cancelled_at.is_(None),
                    Booking.end_at > self.clock(),
                )
                .order_by(Booking.start_at)
            )
            return [self.describe(booking) for booking in bookings]
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `46 passed`, lint clean.

- [ ] **Step 5: Commit**

```bash
git add app/booking.py tests/test_booking.py
git commit -m "Add room, availability, schedule and my-bookings queries

The assistant had no way to answer 'what is free'. Schedules are compact ranges clipped to
business hours per day, so 'free' always means bookable; other users' bookings show only
as occupied so their titles never reach another user's LLM context."
```

---

### Task 6: Authentication

**Files:**
- Create: `app/auth.py`
- Modify: `app/main.py` (whole file), `tests/conftest.py`, `tests/support.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `password_hasher`, `create_session_factory`, `seed`, `BookingService`.
- Produces: `CurrentUser(id: int, username: str)` (frozen dataclass); `authenticate(session_factory, username, password) -> CurrentUser | None`; `create_access_token(user, secret, lifetime: timedelta) -> str`; `read_access_token(token, secret) -> CurrentUser | None`; `create_app(settings=None, clock=None)` with `app.state.service`; routes `POST /auth/login` (form fields `username`, `password` → `{"access_token", "token_type", "username"}`), `GET /auth/me`. Test support: `USER1`, `USER2`, `login_headers(client, username) -> dict`; fixtures `user1_headers`, `user2_headers`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/support.py` (add the import at the top: `from app.auth import CurrentUser`):
```python
USER1 = CurrentUser(id=USER1_ID, username="User1")
USER2 = CurrentUser(id=USER2_ID, username="User2")


def login_headers(client, username: str) -> dict:
    response = client.post(
        "/auth/login", data={"username": username, "password": "TechnicalChallengePromtior"}
    )
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
```

In `tests/conftest.py` change the import to `from tests.support import FIXED_NOW, FakeClock, login_headers` and replace the `client` fixture with:
```python
@pytest.fixture
def client(settings, clock) -> TestClient:
    return TestClient(create_app(settings, clock=clock))


@pytest.fixture
def user1_headers(client) -> dict:
    return login_headers(client, "User1")


@pytest.fixture
def user2_headers(client) -> dict:
    return login_headers(client, "User2")
```

`tests/test_api.py` (whole file):
```python
from datetime import timedelta

import pytest

from app.auth import CurrentUser, create_access_token


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_login_returns_a_bearer_token(client):
    response = client.post(
        "/auth/login", data={"username": "User1", "password": "TechnicalChallengePromtior"}
    )
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert response.json()["username"] == "User1"


def test_username_is_case_insensitive(client):
    response = client.post(
        "/auth/login", data={"username": "user1", "password": "TechnicalChallengePromtior"}
    )
    assert response.status_code == 200
    assert response.json()["username"] == "User1"


@pytest.mark.parametrize(
    ("username", "password"), [("User1", "wrong"), ("Nobody", "TechnicalChallengePromtior")]
)
def test_wrong_credentials_are_rejected(client, username, password):
    response = client.post("/auth/login", data={"username": username, "password": password})
    assert response.status_code == 401


def test_me_needs_a_token(client):
    assert client.get("/auth/me").status_code == 401


def test_me_returns_the_logged_in_user(client, user1_headers):
    assert client.get("/auth/me", headers=user1_headers).json() == {"id": 1, "username": "User1"}


@pytest.mark.parametrize(
    ("secret", "lifetime"),
    [("test-secret", timedelta(minutes=-1)), ("another-secret", timedelta(minutes=5))],
)
def test_expired_or_foreign_tokens_are_rejected(client, secret, lifetime):
    token = create_access_token(CurrentUser(id=1, username="User1"), secret, lifetime)
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q`
Expected: `ModuleNotFoundError: No module named 'app.auth'`.

- [ ] **Step 3: Write `app/auth.py` and the new `app/main.py`**

`app/auth.py`:
```python
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db import password_hasher
from app.models import User

# Verifying against a throwaway hash keeps the response time equal for unknown usernames.
HASH_FOR_UNKNOWN_USERS = password_hasher.hash("unknown-user")


@dataclass(frozen=True)
class CurrentUser:
    id: int
    username: str


def authenticate(
    session_factory: sessionmaker[Session], username: str, password: str
) -> CurrentUser | None:
    with session_factory() as session:
        user = session.scalar(select(User).where(func.lower(User.username) == username.lower()))
    if user is None:
        password_hasher.verify(password, HASH_FOR_UNKNOWN_USERS)
        return None
    if not password_hasher.verify(password, user.password_hash):
        return None
    return CurrentUser(id=user.id, username=user.username)


def create_access_token(user: CurrentUser, secret: str, lifetime: timedelta) -> str:
    claims = {"sub": str(user.id), "username": user.username, "exp": datetime.now(UTC) + lifetime}
    return jwt.encode(claims, secret, algorithm="HS256")


def read_access_token(token: str, secret: str) -> CurrentUser | None:
    try:
        claims = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        return None
    return CurrentUser(id=int(claims["sub"]), username=claims["username"])
```

`app/main.py` (whole file):
```python
import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.auth import CurrentUser, authenticate, create_access_token, read_access_token
from app.booking import BookingService
from app.config import Settings
from app.db import create_session_factory, seed


def system_clock() -> datetime:
    return datetime.now(UTC)


def create_app(settings: Settings | None = None, clock=None) -> FastAPI:
    if settings is None:
        settings = Settings()
    if clock is None:
        clock = system_clock
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    session_factory = create_session_factory(settings.sqlalchemy_database_url)
    seed(session_factory, settings.seed_password)
    service = BookingService(session_factory, settings, clock)
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

    app = FastAPI(title="Room Booking Assistant")
    app.state.service = service

    def current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> CurrentUser:
        user = read_access_token(token, settings.jwt_secret)
        if user is None:
            raise HTTPException(
                401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
            )
        return user

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/auth/login")
    def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> dict:
        user = authenticate(session_factory, form.username, form.password)
        if user is None:
            raise HTTPException(
                401, "Wrong username or password", headers={"WWW-Authenticate": "Bearer"}
            )
        lifetime = timedelta(minutes=settings.token_lifetime_minutes)
        return {
            "access_token": create_access_token(user, settings.jwt_secret, lifetime),
            "token_type": "bearer",
            "username": user.username,
        }

    @app.get("/auth/me")
    def me(user: Annotated[CurrentUser, Depends(current_user)]) -> dict:
        return {"id": user.id, "username": user.username}

    return app
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `54 passed`, lint clean.

- [ ] **Step 5: Commit**

```bash
git add app/auth.py app/main.py tests
git commit -m "Add login for User1 and User2 with JWT

Every booking must belong to the logged-in user, and nothing identified one. Passwords are
Argon2 hashes, the token carries the user id, unknown usernames cost the same time as wrong
passwords, and the form login makes the Swagger 'Authorize' button work for reviewers."
```

---

### Task 7: Read-only REST API

**Files:**
- Modify: `app/main.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `BookingService.rooms`, `room_schedule`, `bookings_of`, `parse_local_datetime`; `current_user` dependency.
- Produces: `GET /rooms`, `GET /rooms/{room_id}/schedule?start=&end=`, `GET /bookings/mine`; `BookingError` → 404 when the code ends in `NOT_FOUND`, else 400, body `error.as_dict()`.

- [ ] **Step 1: Append the failing tests to `tests/test_api.py`**

Add to the imports: `from tests.support import USER1_ID, USER2_ID, booking_request`. Append:
```python
def test_rooms_need_a_token(client):
    assert client.get("/rooms").status_code == 401


def test_rooms_list_the_capacities(client, user1_headers):
    rooms = client.get("/rooms", headers=user1_headers).json()
    assert len(rooms) == 5
    assert rooms[0] == {"room": "A", "capacity": 4}


def test_schedule_shows_other_users_bookings_as_occupied_only(client, service, user1_headers):
    service.create(USER2_ID, booking_request(title="Secret merger talks"))
    response = client.get(
        "/rooms/b/schedule",
        params={"start": "2026-09-23T10:00", "end": "2026-09-23T11:30"},
        headers=user1_headers,
    )
    assert response.json()["ranges"] == [
        {"start": "2026-09-23T10:00", "end": "2026-09-23T11:30", "status": "occupied", "mine": False}
    ]


def test_schedule_of_an_unknown_room_is_not_found(client, user1_headers):
    response = client.get(
        "/rooms/Z/schedule",
        params={"start": "2026-09-23T08:00", "end": "2026-09-23T20:00"},
        headers=user1_headers,
    )
    assert response.status_code == 404
    assert response.json()["error"] == "ROOM_NOT_FOUND"


def test_schedule_with_unreadable_dates_is_a_bad_request(client, user1_headers):
    response = client.get(
        "/rooms/B/schedule", params={"start": "tomorrow", "end": "later"}, headers=user1_headers
    )
    assert response.status_code == 400
    assert response.json()["error"] == "INVALID_DATETIME"


def test_my_bookings_lists_only_mine(client, service, user1_headers):
    service.create(USER1_ID, booking_request(title="Mine"))
    service.create(USER2_ID, booking_request(room_id="C", title="Theirs", attendees=2))
    bookings = client.get("/bookings/mine", headers=user1_headers).json()
    assert [booking["title"] for booking in bookings] == ["Mine"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_api.py -q`
Expected: the new tests fail with `404 != 401`-style assertion errors (routes missing).

- [ ] **Step 3: Implement in `app/main.py`**

Change the imports:
```python
from fastapi.responses import JSONResponse

from app.booking import BookingError, BookingService
```
Add after the `me` route, before `return app`:
```python
    @app.exception_handler(BookingError)
    def booking_error_response(request, error: BookingError) -> JSONResponse:
        if error.code.endswith("NOT_FOUND"):
            return JSONResponse(error.as_dict(), status_code=404)
        return JSONResponse(error.as_dict(), status_code=400)

    @app.get("/rooms")
    def list_rooms(user: Annotated[CurrentUser, Depends(current_user)]) -> list[dict]:
        return service.rooms()

    @app.get("/rooms/{room_id}/schedule")
    def room_schedule(
        room_id: str, start: str, end: str, user: Annotated[CurrentUser, Depends(current_user)]
    ) -> dict:
        return service.room_schedule(
            user.id,
            room_id.upper(),
            service.parse_local_datetime(start),
            service.parse_local_datetime(end),
        )

    @app.get("/bookings/mine")
    def my_bookings(user: Annotated[CurrentUser, Depends(current_user)]) -> list[dict]:
        return service.bookings_of(user.id)
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `60 passed`, lint clean.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "Add read-only REST endpoints over the booking service

The web page needs the schedule without going through the LLM, and reviewers can check the
rules from /docs. Booking errors keep their code and data; unknown rooms are 404."
```

---

### Task 8: The agent — tools, prompt, confirmation gate

**Files:**
- Create: `app/agent.py`, `tests/test_agent.py`
- Modify: `tests/support.py`

**Interfaces:**
- Consumes: `BookingService` (all methods and attributes), `BookingError`, `BookingRequest`, `CurrentUser`, `Settings`.
- Produces: `SYSTEM_PROMPT: str`; `build_openai_model(settings) -> ChatOpenAI`; `booking_request_from(service, arguments: dict) -> BookingRequest`; `describe_time_range(service, start, end) -> str` (`"Wed 23 Sep, 10:00–11:30"`); `booking_errors_as_tool_results` middleware; `build_tools(service) -> list` (tools `list_available_rooms`, `get_room_schedule`, `list_my_bookings`, `create_booking`, `cancel_booking`); `build_agent(service, chat_model)` → compiled agent with `context_schema=CurrentUser` and `InMemorySaver`. Confirmation summaries: `"Book room B · Wed 23 Sep, 10:00–11:30 · Interview · 4 attendees"` and `"Cancel booking 1 · room B · Wed 23 Sep, 10:00–11:30 · Interview with John Doe"`. Test support: `ScriptedChatModel` (records `received`), `scripted(*messages)`, `tool_call(name, **arguments) -> AIMessage`.

- [ ] **Step 1: Add the scripted model to `tests/support.py`**

Add to the imports:
```python
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import Field
```
Append:
```python
class ScriptedChatModel(GenericFakeChatModel):
    received: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted(*messages: AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter(messages))


def tool_call(name: str, **arguments) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": arguments, "id": f"call_{uuid4().hex[:8]}"}]
    )
```

- [ ] **Step 2: Write the failing tests**

`tests/test_agent.py`:
```python
import json

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from app.agent import build_agent, build_tools
from tests.support import USER1, USER2_ID, booking_request, scripted, tool_call

APPROVE = Command(resume={"decisions": [{"type": "approve"}]})
REJECT = Command(resume={"decisions": [{"type": "reject"}]})
CREATE_B = {
    "room": "b",
    "start": "2026-09-23T10:00",
    "end": "2026-09-23T11:30",
    "title": "Interview",
    "attendees": 4,
}


def run(agent, agent_input):
    return agent.invoke(
        agent_input, {"configurable": {"thread_id": "test"}}, context=USER1, version="v2"
    )


def ask(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def tool_results(result) -> list[dict]:
    return [
        json.loads(message.content)
        for message in result.value["messages"]
        if isinstance(message, ToolMessage)
    ]


def test_tools_never_expose_the_user_to_the_model(service):
    for agent_tool in build_tools(service):
        properties = agent_tool.tool_call_schema.model_json_schema().get("properties", {})
        assert "runtime" not in properties
        assert [name for name in properties if "user" in name] == []


def test_valid_booking_pauses_and_approval_writes_it(service):
    agent = build_agent(
        service, scripted(tool_call("create_booking", **CREATE_B), AIMessage(content="Booked."))
    )
    paused = run(agent, ask("book B"))
    [action] = paused.interrupts[0].value["action_requests"]
    assert action["description"] == "Book room B · Wed 23 Sep, 10:00–11:30 · Interview · 4 attendees"
    assert service.bookings_of(USER1.id) == []

    finished = run(agent, APPROVE)
    assert finished.interrupts == ()
    assert [booking["room"] for booking in service.bookings_of(USER1.id)] == ["B"]
    assert finished.value["messages"][-1].text == "Booked."


def test_invalid_booking_does_not_pause_and_writes_nothing(service):
    too_many = {**CREATE_B, "room": "A", "attendees": 30}
    agent = build_agent(
        service, scripted(tool_call("create_booking", **too_many), AIMessage(content="Too small."))
    )
    result = run(agent, ask("book A for 30"))
    assert result.interrupts == ()
    assert tool_results(result)[0]["error"] == "CAPACITY_EXCEEDED"
    assert service.bookings_of(USER1.id) == []


def test_rejection_writes_nothing(service):
    agent = build_agent(
        service, scripted(tool_call("create_booking", **CREATE_B), AIMessage(content="Ok."))
    )
    run(agent, ask("book B"))
    run(agent, REJECT)
    assert service.bookings_of(USER1.id) == []


def test_cancelling_someone_elses_booking_does_not_pause(service):
    theirs = service.create(USER2_ID, booking_request())
    agent = build_agent(
        service,
        scripted(
            tool_call("cancel_booking", booking_id=theirs["booking_id"]),
            AIMessage(content="Not yours."),
        ),
    )
    result = run(agent, ask(f"cancel booking {theirs['booking_id']}"))
    assert result.interrupts == ()
    assert tool_results(result)[0]["error"] == "NOT_OWNER"
    assert service.bookings_of(USER2_ID) == [theirs]


def test_cancelling_my_booking_pauses_with_its_summary(service):
    mine = service.create(USER1.id, booking_request())
    agent = build_agent(
        service,
        scripted(
            tool_call("cancel_booking", booking_id=mine["booking_id"]),
            AIMessage(content="Cancelled."),
        ),
    )
    paused = run(agent, ask("cancel it"))
    [action] = paused.interrupts[0].value["action_requests"]
    assert action["description"] == (
        "Cancel booking 1 · room B · Wed 23 Sep, 10:00–11:30 · Interview with John Doe"
    )
    run(agent, APPROVE)
    assert service.bookings_of(USER1.id) == []


def test_system_prompt_carries_the_user_and_the_office_time(service):
    model = scripted(AIMessage(content="Hi!"))
    run(build_agent(service, model), ask("hello"))
    system_message = model.received[0][0]
    assert system_message.type == "system"
    assert "User1" in system_message.text
    assert "2026-09-22T09:00 (Tuesday)" in system_message.text
    assert "A (4 people)" in system_message.text
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_agent.py -q`
Expected: `ModuleNotFoundError: No module named 'app.agent'`.

- [ ] **Step 4: Write `app/agent.py`**

```python
import json
import logging
from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app.auth import CurrentUser
from app.booking import BookingError, BookingRequest, BookingService
from app.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the meeting room assistant of the Promtior office at Cubo Itaú, Montevideo. Your only job is to help {username} check room availability, book rooms and cancel their own bookings.

Now it is {now} ({weekday}) in {timezone}. Resolve relative dates such as "tomorrow" or "next Friday" from this moment. Pass times to tools as local ISO 8601 without offset, for example 2026-09-23T10:00.

Rooms and capacities: {rooms}.

Rules the booking system enforces:
- Bookings start and end on the hour or the half hour, last at most 3 hours, and fit between {opening} and {closing} on a single day.
- The attendees must fit in the room. Every booking needs a title.
- A room holds one booking at a time.

How to work:
- To book you need the room, the date, the start time, the end time or duration, the title and the number of attendees. Ask for whatever is missing. Never make up a title or a number of attendees.
- If the user has no room preference, check availability and suggest the smallest free room that fits.
- Do not ask for confirmation in text before creating or cancelling: the system shows the user a confirmation card by itself.
- When a tool answers with "ok": false, explain the reason in plain words and offer the closest alternative, such as another time or another room.
- To cancel, find the booking id with list_my_bookings. Never guess an id. Users can only cancel their own bookings.
- Other people's bookings appear only as occupied. Do not speculate about them.
- Politely decline anything that is not about meeting room bookings.
- Answer in the language the user writes in, briefly, in plain text without Markdown."""


def build_openai_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        use_responses_api=True,
        reasoning={"effort": settings.openai_reasoning_effort},
        timeout=30,
        max_retries=2,
    )


def booking_request_from(service: BookingService, arguments: dict) -> BookingRequest:
    return BookingRequest(
        room_id=arguments["room"].strip().upper(),
        start=service.parse_local_datetime(arguments["start"]),
        end=service.parse_local_datetime(arguments["end"]),
        title=arguments["title"],
        attendees=arguments["attendees"],
    )


def describe_time_range(service: BookingService, start: datetime, end: datetime) -> str:
    local_start = start.astimezone(service.timezone)
    local_end = end.astimezone(service.timezone)
    return (
        f"{local_start:%a} {local_start.day} {local_start:%b}, "
        f"{local_start:%H:%M}–{local_end:%H:%M}"
    )


@wrap_tool_call
def booking_errors_as_tool_results(request, handler):
    try:
        return handler(request)
    except BookingError as error:
        logger.info("Tool %s refused: %s", request.tool_call["name"], error.code)
        result = error.as_dict()
    except Exception:
        logger.exception("Tool %s failed", request.tool_call["name"])
        result = {
            "ok": False,
            "error": "INTERNAL_ERROR",
            "message": "The booking system failed. Ask the user to try again in a moment.",
        }
    return ToolMessage(
        content=json.dumps(result),
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
        status="error",
    )


def build_tools(service: BookingService) -> list:
    @tool(parse_docstring=True)
    def list_available_rooms(start: str, end: str, attendees: int | None = None) -> dict:
        """List the rooms that are free for the whole time range.

        Args:
            start: Local start time, ISO 8601 without offset, for example 2026-09-23T10:00.
            end: Local end time, ISO 8601 without offset.
            attendees: Keep only rooms with at least this capacity.
        """
        rooms = service.available_rooms(
            service.parse_local_datetime(start), service.parse_local_datetime(end), attendees
        )
        return {"ok": True, "rooms": rooms}

    @tool(parse_docstring=True)
    def get_room_schedule(
        room: str, start: str, end: str, runtime: ToolRuntime[CurrentUser]
    ) -> dict:
        """Show which parts of a time range are free or occupied in one room.

        Args:
            room: Room letter, A to E.
            start: Local start time, ISO 8601 without offset.
            end: Local end time, ISO 8601 without offset.
        """
        schedule = service.room_schedule(
            runtime.context.id,
            room.strip().upper(),
            service.parse_local_datetime(start),
            service.parse_local_datetime(end),
        )
        return {"ok": True, **schedule}

    @tool
    def list_my_bookings(runtime: ToolRuntime[CurrentUser]) -> dict:
        """List the current user's upcoming bookings with their ids."""
        return {"ok": True, "bookings": service.bookings_of(runtime.context.id)}

    @tool(parse_docstring=True)
    def create_booking(
        room: str,
        start: str,
        end: str,
        title: str,
        attendees: int,
        runtime: ToolRuntime[CurrentUser],
    ) -> dict:
        """Book a room for the current user. The system asks the user to confirm before it runs.

        Args:
            room: Room letter, A to E.
            start: Local start time, ISO 8601 without offset, on the hour or half hour.
            end: Local end time, ISO 8601 without offset, on the hour or half hour.
            title: Title of the meeting, as the user said it.
            attendees: Number of people attending.
        """
        arguments = {"room": room, "start": start, "end": end, "title": title, "attendees": attendees}
        booking = service.create(runtime.context.id, booking_request_from(service, arguments))
        return {"ok": True, "booking": booking}

    @tool(parse_docstring=True)
    def cancel_booking(booking_id: int, runtime: ToolRuntime[CurrentUser]) -> dict:
        """Cancel one of the current user's bookings. The system asks the user to confirm before it runs.

        Args:
            booking_id: Id of the booking, from list_my_bookings.
        """
        return {"ok": True, "cancelled": service.cancel(runtime.context.id, booking_id)}

    return [list_available_rooms, get_room_schedule, list_my_bookings, create_booking, cancel_booking]


def build_agent(service: BookingService, chat_model):
    rooms_text = ", ".join(f"{room['room']} ({room['capacity']} people)" for room in service.rooms())

    @dynamic_prompt
    def system_prompt(request: ModelRequest) -> str:
        now = service.clock().astimezone(service.timezone)
        return SYSTEM_PROMPT.format(
            username=request.runtime.context.username,
            now=now.strftime("%Y-%m-%dT%H:%M"),
            weekday=now.strftime("%A"),
            timezone=service.timezone.key,
            rooms=rooms_text,
            opening=f"{service.opening_hour:02d}:00",
            closing=f"{service.closing_hour:02d}:00",
        )

    def create_would_succeed(request) -> bool:
        try:
            service.check_create(booking_request_from(service, request.tool_call["args"]))
        except (BookingError, KeyError, AttributeError, TypeError):
            return False
        return True

    def cancel_would_succeed(request) -> bool:
        try:
            service.check_cancel(
                request.runtime.context.id, int(request.tool_call["args"]["booking_id"])
            )
        except (BookingError, KeyError, TypeError, ValueError):
            return False
        return True

    def describe_create(tool_call, state, runtime) -> str:
        request = booking_request_from(service, tool_call["args"])
        attendees = f"{request.attendees} attendees"
        if request.attendees == 1:
            attendees = "1 attendee"
        time_range = describe_time_range(service, request.start, request.end)
        return f"Book room {request.room_id} · {time_range} · {request.title.strip()} · {attendees}"

    def describe_cancel(tool_call, state, runtime) -> str:
        booking = service.check_cancel(runtime.context.id, int(tool_call["args"]["booking_id"]))
        time_range = describe_time_range(
            service,
            service.parse_local_datetime(booking["start"]),
            service.parse_local_datetime(booking["end"]),
        )
        return (
            f"Cancel booking {booking['booking_id']} · room {booking['room']} · "
            f"{time_range} · {booking['title']}"
        )

    confirmation_gate = HumanInTheLoopMiddleware(
        interrupt_on={
            "create_booking": {
                "allowed_decisions": ["approve", "reject"],
                "when": create_would_succeed,
                "description": describe_create,
            },
            "cancel_booking": {
                "allowed_decisions": ["approve", "reject"],
                "when": cancel_would_succeed,
                "description": describe_cancel,
            },
        }
    )
    return create_agent(
        chat_model,
        tools=build_tools(service),
        middleware=[
            system_prompt,
            booking_errors_as_tool_results,
            confirmation_gate,
            ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),
        ],
        context_schema=CurrentUser,
        checkpointer=InMemorySaver(),
    )
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `67 passed`, lint clean.

- [ ] **Step 6: Smoke test against the real model**

The author creates `.env` from `.env.example` with a key from a dedicated OpenAI project that has a monthly budget, and a random `JWT_SECRET` (`uv run python -c "import secrets; print(secrets.token_urlsafe(48))"`). Check it is ignored: `git check-ignore -v .env` must print a match.

Write this throwaway script to the session scratchpad (not the repo) as `smoke_agent.py`:
```python
import tempfile
from pathlib import Path

from app.agent import build_agent, build_openai_model
from app.booking import BookingService
from app.config import Settings
from app.db import create_session_factory, seed
from tests.support import FIXED_NOW, USER1, FakeClock

database = Path(tempfile.mkdtemp()) / "smoke.db"
settings = Settings(database_url=f"sqlite:///{database.as_posix()}")
session_factory = create_session_factory(settings.sqlalchemy_database_url)
seed(session_factory, settings.seed_password)
service = BookingService(session_factory, settings, FakeClock(FIXED_NOW))
agent = build_agent(service, build_openai_model(settings))
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Which rooms are free tomorrow from 10 to 11 for 6 people?"}]},
    {"configurable": {"thread_id": "smoke"}},
    context=USER1,
    version="v2",
)
for message in result.value["messages"]:
    print(type(message).__name__, getattr(message, "tool_calls", ""), message.text[:300])
```
Run: `PYTHONPATH=. uv run python "<scratchpad>/smoke_agent.py"`
Expected: an `AIMessage` with a `list_available_rooms` call (`start` 2026-09-23T10:00, `attendees` 6), a `ToolMessage` listing C, D and E, and a final `AIMessage` naming them. If OpenAI rejects the `reasoning` parameter for this model, remove that line from `build_openai_model`, rerun, and write the reason in the commit message.

- [ ] **Step 7: Commit**

```bash
git add app/agent.py tests/support.py tests/test_agent.py
git commit -m "Add the booking agent with a confirmation gate

The chatbot did not exist. Five tools delegate to BookingService; the user reaches them only
through runtime context, so no prompt can book or cancel as someone else. Writes pause for
confirmation only when a dry run says they will succeed, so the user never confirms a
booking that then fails; domain errors come back as structured tool results."
```

---

### Task 9: Chat API — conversations, decisions, limits

**Files:**
- Create: `app/chat.py`, `tests/test_chat.py`
- Modify: `app/main.py` (whole file), `tests/conftest.py`

**Interfaces:**
- Consumes: `build_agent`, `build_openai_model`, `CurrentUser`, `Settings.messages_per_window`, `message_window_minutes`.
- Produces: `NothingToDecide` exception; `thread_config(user, conversation_id) -> dict`; `Conversations(agent)` with `send_message(user, conversation_id, message) -> dict`, `decide(user, conversation_id, approve) -> dict`, `reply(conversation_id, result) -> dict` (`{"conversation_id", "reply", "pending_actions": [{"tool", "summary"}]}`); `MessageRateLimiter(limit, window, clock).allow(user_id) -> bool`; `create_app(settings=None, chat_model=None, clock=None)` with `app.state.agent`; `POST /chat/messages` `{conversation_id?, message}`, `POST /chat/decisions` `{conversation_id, approve}`; fixture `make_client(*script)`.

- [ ] **Step 1: Update the fixtures**

`tests/conftest.py` (whole file):
```python
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.booking import BookingService
from app.config import Settings
from app.db import create_session_factory, seed
from app.main import create_app
from tests.support import FIXED_NOW, FakeClock, login_headers, scripted


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        jwt_secret="test-secret",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(FIXED_NOW)


@pytest.fixture
def service(settings, clock) -> BookingService:
    session_factory = create_session_factory(settings.sqlalchemy_database_url)
    seed(session_factory, settings.seed_password)
    return BookingService(session_factory, settings, clock)


@pytest.fixture
def make_client(settings, clock):
    def make(*script: AIMessage) -> TestClient:
        return TestClient(create_app(settings, chat_model=scripted(*script), clock=clock))

    return make


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client(AIMessage(content="Hello!"))


@pytest.fixture
def user1_headers(client) -> dict:
    return login_headers(client, "User1")


@pytest.fixture
def user2_headers(client) -> dict:
    return login_headers(client, "User2")
```

- [ ] **Step 2: Write the failing tests**

`tests/test_chat.py`:
```python
from datetime import timedelta

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from app.main import create_app
from tests.support import login_headers, scripted, tool_call

CREATE_B = {
    "room": "B",
    "start": "2026-09-23T10:00",
    "end": "2026-09-23T11:30",
    "title": "Interview",
    "attendees": 4,
}


def send(client, headers, message, conversation_id=None):
    return client.post(
        "/chat/messages",
        json={"conversation_id": conversation_id, "message": message},
        headers=headers,
    )


def decide(client, headers, conversation_id, approve=True):
    return client.post(
        "/chat/decisions",
        json={"conversation_id": conversation_id, "approve": approve},
        headers=headers,
    )


def test_a_booking_through_chat_waits_for_confirmation(make_client):
    client = make_client(tool_call("create_booking", **CREATE_B), AIMessage(content="Booked."))
    headers = login_headers(client, "User1")
    paused = send(client, headers, "book B tomorrow 10 to 11:30").json()
    assert paused["pending_actions"] == [
        {
            "tool": "create_booking",
            "summary": "Book room B · Wed 23 Sep, 10:00–11:30 · Interview · 4 attendees",
        }
    ]
    assert client.get("/bookings/mine", headers=headers).json() == []

    done = decide(client, headers, paused["conversation_id"]).json()
    assert done == {"conversation_id": paused["conversation_id"], "reply": "Booked.", "pending_actions": []}
    assert [booking["room"] for booking in client.get("/bookings/mine", headers=headers).json()] == ["B"]


def test_confirming_twice_books_once(make_client):
    client = make_client(tool_call("create_booking", **CREATE_B), AIMessage(content="Booked."))
    headers = login_headers(client, "User1")
    paused = send(client, headers, "book B").json()
    assert decide(client, headers, paused["conversation_id"]).status_code == 200
    assert decide(client, headers, paused["conversation_id"]).status_code == 409
    assert len(client.get("/bookings/mine", headers=headers).json()) == 1


def test_typing_while_a_confirmation_is_pending_rejects_it_with_that_message(make_client):
    client = make_client(
        tool_call("create_booking", **CREATE_B),
        tool_call("create_booking", **{**CREATE_B, "attendees": 6}),
        AIMessage(content="Booked for 6."),
    )
    headers = login_headers(client, "User1")
    first = send(client, headers, "book B for 4").json()
    second = send(client, headers, "make it 6 people", first["conversation_id"]).json()
    assert second["pending_actions"][0]["summary"].endswith("6 attendees")

    state = client.app.state.agent.get_state(
        {"configurable": {"thread_id": f"1:{first['conversation_id']}"}}
    )
    rejections = [m.content for m in state.values["messages"] if isinstance(m, ToolMessage)]
    assert "make it 6 people" in rejections[0]

    decide(client, headers, first["conversation_id"])
    [booking] = client.get("/bookings/mine", headers=headers).json()
    assert booking["attendees"] == 6


def test_conversations_are_private_to_their_user(make_client):
    client = make_client(tool_call("create_booking", **CREATE_B))
    user1 = login_headers(client, "User1")
    user2 = login_headers(client, "User2")
    paused = send(client, user1, "book B").json()
    assert decide(client, user2, paused["conversation_id"]).status_code == 409
    assert client.get("/bookings/mine", headers=user1).json() == []


def test_a_decision_without_a_pending_action_is_a_conflict(client, user1_headers):
    assert decide(client, user1_headers, "nothing-here").status_code == 409


def test_long_messages_are_rejected(client, user1_headers):
    assert send(client, user1_headers, "x" * 1001).status_code == 422


def test_messages_are_rate_limited_per_user(settings, clock):
    limited = settings.model_copy(update={"messages_per_window": 2})
    model = scripted(AIMessage(content="a"), AIMessage(content="b"), AIMessage(content="c"))
    client = TestClient(create_app(limited, chat_model=model, clock=clock))
    headers = login_headers(client, "User1")
    statuses = [send(client, headers, "hi").status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    clock.now += timedelta(minutes=10)
    assert send(client, headers, "hi").status_code == 200
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest -q`
Expected: every test that uses `client` errors with `TypeError: create_app() got an unexpected keyword argument 'chat_model'`.

- [ ] **Step 4: Write `app/chat.py` and the new `app/main.py`**

`app/chat.py`:
```python
from collections import defaultdict, deque
from datetime import timedelta
from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.auth import CurrentUser


class NothingToDecide(Exception):
    pass


def thread_config(user: CurrentUser, conversation_id: str) -> dict:
    return {"configurable": {"thread_id": f"{user.id}:{conversation_id}"}}


class Conversations:
    def __init__(self, agent):
        self.agent = agent

    def send_message(self, user: CurrentUser, conversation_id: str | None, message: str) -> dict:
        if conversation_id is None:
            conversation_id = uuid4().hex
        config = thread_config(user, conversation_id)
        pending = self.agent.get_state(config).interrupts
        if pending:
            action_count = len(pending[0].value["action_requests"])
            rejection = {"type": "reject", "message": message}
            agent_input = Command(resume={"decisions": [rejection] * action_count})
        else:
            agent_input = {"messages": [{"role": "user", "content": message}]}
        result = self.agent.invoke(agent_input, config, context=user, version="v2")
        return self.reply(conversation_id, result)

    def decide(self, user: CurrentUser, conversation_id: str, approve: bool) -> dict:
        config = thread_config(user, conversation_id)
        pending = self.agent.get_state(config).interrupts
        if not pending:
            raise NothingToDecide()
        decision = {"type": "reject"}
        if approve:
            decision = {"type": "approve"}
        action_count = len(pending[0].value["action_requests"])
        result = self.agent.invoke(
            Command(resume={"decisions": [decision] * action_count}),
            config,
            context=user,
            version="v2",
        )
        return self.reply(conversation_id, result)

    def reply(self, conversation_id: str, result) -> dict:
        pending_actions = []
        for interrupt in result.interrupts:
            for action in interrupt.value["action_requests"]:
                pending_actions.append({"tool": action["name"], "summary": action["description"]})
        text = ""
        last_message = result.value["messages"][-1]
        if isinstance(last_message, AIMessage):
            text = last_message.text
        return {"conversation_id": conversation_id, "reply": text, "pending_actions": pending_actions}


class MessageRateLimiter:
    def __init__(self, limit: int, window: timedelta, clock):
        self.limit = limit
        self.window = window
        self.clock = clock
        self.sent_at = defaultdict(deque)

    def allow(self, user_id: int) -> bool:
        now = self.clock()
        sent_at = self.sent_at[user_id]
        while sent_at and now - sent_at[0] >= self.window:
            sent_at.popleft()
        if len(sent_at) >= self.limit:
            return False
        sent_at.append(now)
        return True
```

`app/main.py` (whole file):
```python
import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from app.agent import build_agent, build_openai_model
from app.auth import CurrentUser, authenticate, create_access_token, read_access_token
from app.booking import BookingError, BookingService
from app.chat import Conversations, MessageRateLimiter, NothingToDecide
from app.config import Settings
from app.db import create_session_factory, seed


class MessageIn(BaseModel):
    conversation_id: str | None = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=1000)


class DecisionIn(BaseModel):
    conversation_id: str = Field(max_length=64)
    approve: bool


def system_clock() -> datetime:
    return datetime.now(UTC)


def create_app(settings: Settings | None = None, chat_model=None, clock=None) -> FastAPI:
    if settings is None:
        settings = Settings()
    if clock is None:
        clock = system_clock
    if chat_model is None:
        chat_model = build_openai_model(settings)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    session_factory = create_session_factory(settings.sqlalchemy_database_url)
    seed(session_factory, settings.seed_password)
    service = BookingService(session_factory, settings, clock)
    agent = build_agent(service, chat_model)
    conversations = Conversations(agent)
    rate_limiter = MessageRateLimiter(
        settings.messages_per_window, timedelta(minutes=settings.message_window_minutes), clock
    )
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

    app = FastAPI(title="Room Booking Assistant")
    app.state.service = service
    app.state.agent = agent

    def current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> CurrentUser:
        user = read_access_token(token, settings.jwt_secret)
        if user is None:
            raise HTTPException(
                401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
            )
        return user

    def within_rate_limit(user: Annotated[CurrentUser, Depends(current_user)]) -> CurrentUser:
        if not rate_limiter.allow(user.id):
            raise HTTPException(429, "Too many messages. Try again in a few minutes.")
        return user

    @app.exception_handler(BookingError)
    def booking_error_response(request, error: BookingError) -> JSONResponse:
        if error.code.endswith("NOT_FOUND"):
            return JSONResponse(error.as_dict(), status_code=404)
        return JSONResponse(error.as_dict(), status_code=400)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/auth/login")
    def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> dict:
        user = authenticate(session_factory, form.username, form.password)
        if user is None:
            raise HTTPException(
                401, "Wrong username or password", headers={"WWW-Authenticate": "Bearer"}
            )
        lifetime = timedelta(minutes=settings.token_lifetime_minutes)
        return {
            "access_token": create_access_token(user, settings.jwt_secret, lifetime),
            "token_type": "bearer",
            "username": user.username,
        }

    @app.get("/auth/me")
    def me(user: Annotated[CurrentUser, Depends(current_user)]) -> dict:
        return {"id": user.id, "username": user.username}

    @app.get("/rooms")
    def list_rooms(user: Annotated[CurrentUser, Depends(current_user)]) -> list[dict]:
        return service.rooms()

    @app.get("/rooms/{room_id}/schedule")
    def room_schedule(
        room_id: str, start: str, end: str, user: Annotated[CurrentUser, Depends(current_user)]
    ) -> dict:
        return service.room_schedule(
            user.id,
            room_id.upper(),
            service.parse_local_datetime(start),
            service.parse_local_datetime(end),
        )

    @app.get("/bookings/mine")
    def my_bookings(user: Annotated[CurrentUser, Depends(current_user)]) -> list[dict]:
        return service.bookings_of(user.id)

    @app.post("/chat/messages")
    def send_message(
        body: MessageIn, user: Annotated[CurrentUser, Depends(within_rate_limit)]
    ) -> dict:
        return conversations.send_message(user, body.conversation_id, body.message)

    @app.post("/chat/decisions")
    def decide(body: DecisionIn, user: Annotated[CurrentUser, Depends(within_rate_limit)]) -> dict:
        try:
            return conversations.decide(user, body.conversation_id, body.approve)
        except NothingToDecide:
            raise HTTPException(409, "There is no action waiting for confirmation.") from None

    return app
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `74 passed`, lint clean.

- [ ] **Step 6: Commit**

```bash
git add app/chat.py app/main.py tests/conftest.py tests/test_chat.py
git commit -m "Add the chat API with confirmation decisions and limits

The agent was not reachable over HTTP. Threads are keyed by user and conversation on the
server so nobody can resume another user's thread; typing while a card is pending rejects
it with the text as the reason; a decision without a pending action is 409, so a double
click books once. A per-user message limit protects the OpenAI budget of a public demo."
```

---

### Task 10: Web page

**Files:**
- Create: `app/static/index.html`
- Modify: `app/main.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `POST /auth/login`, `GET /auth/me`, `GET /rooms`, `GET /rooms/{id}/schedule`, `POST /chat/messages`, `POST /chat/decisions`.
- Produces: `GET /` serving the page.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api.py`:
```python
def test_the_web_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>Room Booking Assistant</title>" in response.text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_api.py::test_the_web_page_is_served -q`
Expected: FAIL, `assert 404 == 200`.

- [ ] **Step 3: Serve the page from `app/main.py`**

Add the imports `from pathlib import Path` and `from fastapi.responses import FileResponse, JSONResponse` (replacing the `JSONResponse`-only import), add below the imports:
```python
INDEX_PAGE = Path(__file__).parent / "static" / "index.html"
```
and add this route right before the `/health` route:
```python
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_PAGE)
```

- [ ] **Step 4: Write `app/static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Room Booking Assistant</title>
<style>
  :root {
    --bg: #f6f7fb; --panel: #ffffff; --text: #1d2130; --muted: #6b7185; --border: #e3e6ef;
    --accent: #4f46e5; --accent-text: #ffffff; --free: #eefbf3; --busy: #e4e7ef;
    --mine: #e0e7ff; --mine-border: #818cf8; --danger: #b42318;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #11131a; --panel: #1a1d27; --text: #e7e9f1; --muted: #9aa0b4; --border: #2a2e3b;
      --accent: #818cf8; --accent-text: #11131a; --free: #15271d; --busy: #2b2f3c;
      --mine: #272c52; --mine-border: #818cf8; --danger: #f97066;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
  header { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 20px; border-bottom: 1px solid var(--border); background: var(--panel); }
  header h1 { font-size: 16px; margin: 0; }
  .who { color: var(--muted); font-size: 14px; margin-right: 8px; }
  button { font: inherit; border: 1px solid var(--border); background: var(--panel); color: var(--text); border-radius: 8px; padding: 8px 14px; cursor: pointer; }
  button.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-text); }
  button:disabled { opacity: .5; cursor: default; }
  input, textarea { font: inherit; color: var(--text); background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 9px 11px; }
  .hidden { display: none !important; }
  #login { max-width: 360px; margin: 12vh auto; padding: 28px; background: var(--panel); border: 1px solid var(--border); border-radius: 14px; display: grid; gap: 12px; }
  #login h2 { margin: 0; font-size: 18px; }
  #login p { margin: 0; color: var(--muted); font-size: 14px; }
  .error { color: var(--danger); font-size: 14px; min-height: 1em; }
  main { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 16px; padding: 16px; height: calc(100vh - 58px); }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; height: auto; } }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; display: flex; flex-direction: column; min-height: 0; }
  #messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; min-height: 320px; }
  .bubble { max-width: 85%; padding: 10px 13px; border-radius: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .user { align-self: flex-end; background: var(--accent); color: var(--accent-text); }
  .assistant { align-self: flex-start; background: var(--bg); border: 1px solid var(--border); }
  .thinking { color: var(--muted); font-style: italic; }
  .card { align-self: flex-start; max-width: 85%; border: 1px solid var(--mine-border); background: var(--mine); border-radius: 12px; padding: 12px 14px; display: grid; gap: 10px; }
  .card .actions { display: flex; gap: 8px; }
  .suggestions { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 16px 10px; }
  .suggestions button { font-size: 13px; padding: 5px 10px; border-radius: 999px; color: var(--muted); }
  #composer { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid var(--border); }
  #composer textarea { flex: 1; resize: none; height: 44px; }
  .schedule-head { display: flex; align-items: center; justify-content: space-between; padding: 14px 16px; border-bottom: 1px solid var(--border); }
  .schedule-head h2 { font-size: 15px; margin: 0; }
  .legend { display: flex; gap: 14px; padding: 12px 16px 0; color: var(--muted); font-size: 12px; }
  .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; background: var(--swatch); }
  #grid-wrap { overflow: auto; padding: 10px 16px 16px; }
  table { border-collapse: separate; border-spacing: 3px; width: 100%; font-size: 12px; }
  th { color: var(--muted); font-weight: 600; }
  td.time { color: var(--muted); text-align: right; padding-right: 6px; white-space: nowrap; }
  td.slot { height: 18px; border-radius: 4px; background: var(--free); }
  td.slot.occupied { background: var(--busy); }
  td.slot.mine { background: var(--mine); box-shadow: inset 0 0 0 1px var(--mine-border); }
</style>
</head>
<body>
<header>
  <h1>Cubo Itaú · Meeting rooms</h1>
  <div><span class="who" id="who"></span><button id="logout" class="hidden" type="button">Sign out</button></div>
</header>

<form id="login">
  <h2>Sign in</h2>
  <p>Use User1 or User2.</p>
  <input id="username" autocomplete="username" placeholder="Username" required>
  <input id="password" type="password" autocomplete="current-password" placeholder="Password" required>
  <button class="primary" type="submit">Sign in</button>
  <div class="error" id="login-error"></div>
</form>

<main id="app" class="hidden">
  <section class="panel">
    <div id="messages"></div>
    <div class="suggestions" id="suggestions">
      <button type="button">Which rooms are free tomorrow from 10:00 to 11:00?</button>
      <button type="button">Book room B tomorrow from 15:00 to 16:00 for 4 people, title "Design review"</button>
      <button type="button">Show my bookings</button>
    </div>
    <form id="composer">
      <textarea id="input" placeholder="Ask to check, book or cancel a room…" maxlength="1000"></textarea>
      <button class="primary" type="submit" id="send">Send</button>
    </form>
  </section>
  <section class="panel">
    <div class="schedule-head">
      <h2>Schedule</h2>
      <input type="date" id="day">
    </div>
    <div class="legend">
      <span style="--swatch: var(--free)">Free</span>
      <span style="--swatch: var(--busy)">Occupied</span>
      <span style="--swatch: var(--mine)">Yours</span>
    </div>
    <div id="grid-wrap"><table id="grid"></table></div>
  </section>
</main>

<script>
const state = { token: null, username: null, conversationId: null, busy: false };
const byId = (id) => document.getElementById(id);

function remember(key, value) {
  try {
    if (value === null) sessionStorage.removeItem(key);
    else sessionStorage.setItem(key, value);
  } catch (error) {}
}

function recall(key) {
  try { return sessionStorage.getItem(key); } catch (error) { return null; }
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401 && state.token) {
    signOut();
    throw new Error("Your session expired. Sign in again.");
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    let message = body.message || `Request failed (${response.status})`;
    if (typeof body.detail === "string") message = body.detail;
    throw new Error(message);
  }
  return body;
}

function showApp() {
  byId("login").classList.add("hidden");
  byId("app").classList.remove("hidden");
  byId("logout").classList.remove("hidden");
  byId("who").textContent = state.username;
  if (!byId("messages").children.length) {
    addBubble("assistant", `Hi ${state.username}! I can check availability, book rooms A–E and cancel your bookings.`);
  }
  loadSchedule();
}

function signOut() {
  state.token = null;
  state.username = null;
  state.conversationId = null;
  remember("token", null);
  remember("username", null);
  byId("messages").replaceChildren();
  byId("app").classList.add("hidden");
  byId("logout").classList.add("hidden");
  byId("login").classList.remove("hidden");
  byId("who").textContent = "";
}

byId("login").addEventListener("submit", async (event) => {
  event.preventDefault();
  byId("login-error").textContent = "";
  const form = new URLSearchParams({ username: byId("username").value, password: byId("password").value });
  try {
    const body = await api("/auth/login", { method: "POST", body: form });
    state.token = body.access_token;
    state.username = body.username;
    remember("token", state.token);
    remember("username", state.username);
    showApp();
  } catch (error) {
    byId("login-error").textContent = error.message;
  }
});

byId("logout").addEventListener("click", signOut);

function scrollToBottom() {
  byId("messages").scrollTop = byId("messages").scrollHeight;
}

function addBubble(kind, text) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${kind}`;
  bubble.textContent = text;
  byId("messages").append(bubble);
  scrollToBottom();
  return bubble;
}

function addConfirmation(actions) {
  const card = document.createElement("div");
  card.className = "card";
  for (const action of actions) {
    const line = document.createElement("div");
    line.textContent = action.summary;
    card.append(line);
  }
  const buttons = document.createElement("div");
  buttons.className = "actions";
  const confirm = document.createElement("button");
  confirm.className = "primary";
  confirm.type = "button";
  confirm.textContent = "Confirm";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.textContent = "Cancel";
  confirm.addEventListener("click", () => decide(true));
  cancel.addEventListener("click", () => decide(false));
  buttons.append(confirm, cancel);
  card.append(buttons);
  byId("messages").append(card);
  scrollToBottom();
}

function closePendingCards() {
  for (const button of document.querySelectorAll(".card button")) button.disabled = true;
}

async function talk(path, payload) {
  state.busy = true;
  byId("send").disabled = true;
  const thinking = addBubble("assistant thinking", "Thinking…");
  try {
    const reply = await api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.conversationId = reply.conversation_id;
    thinking.remove();
    if (reply.reply) addBubble("assistant", reply.reply);
    if (reply.pending_actions.length) addConfirmation(reply.pending_actions);
    loadSchedule();
  } catch (error) {
    thinking.remove();
    addBubble("assistant", error.message);
  } finally {
    state.busy = false;
    byId("send").disabled = false;
  }
}

function decide(approve) {
  if (state.busy) return;
  closePendingCards();
  talk("/chat/decisions", { conversation_id: state.conversationId, approve });
}

function send(text) {
  const message = text.trim();
  if (!message || state.busy) return;
  closePendingCards();
  addBubble("user", message);
  talk("/chat/messages", { conversation_id: state.conversationId, message });
}

byId("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  send(byId("input").value);
  byId("input").value = "";
});

byId("input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    byId("composer").requestSubmit();
  }
});

for (const suggestion of byId("suggestions").querySelectorAll("button")) {
  suggestion.addEventListener("click", () => send(suggestion.textContent));
}

function officeToday() {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/Montevideo" }).format(new Date());
}

function addHalfHour(text) {
  const [day, clock] = text.split("T");
  let [hours, minutes] = clock.split(":").map(Number);
  minutes += 30;
  if (minutes === 60) {
    hours += 1;
    minutes = 0;
  }
  return `${day}T${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;
}

async function loadSchedule() {
  if (!byId("day").value) byId("day").value = officeToday();
  const day = byId("day").value;
  try {
    const rooms = await api("/rooms");
    const schedules = await Promise.all(
      rooms.map((room) => api(`/rooms/${room.room}/schedule?start=${day}T00:00&end=${day}T23:59`))
    );
    renderGrid(rooms, schedules);
  } catch (error) {
    byId("grid").replaceChildren();
  }
}

function renderGrid(rooms, schedules) {
  const table = byId("grid");
  table.replaceChildren();
  const ranges = schedules[0].ranges;
  if (!ranges.length) return;
  const head = table.insertRow();
  head.append(document.createElement("th"));
  for (const room of rooms) {
    const cell = document.createElement("th");
    cell.textContent = `${room.room} · ${room.capacity}`;
    head.append(cell);
  }
  const end = ranges[ranges.length - 1].end;
  for (let time = ranges[0].start; time < end; time = addHalfHour(time)) {
    const row = table.insertRow();
    const label = row.insertCell();
    label.className = "time";
    label.textContent = time.slice(11);
    for (const schedule of schedules) {
      const cell = row.insertCell();
      const range = schedule.ranges.find((candidate) => candidate.start <= time && time < candidate.end);
      cell.className = "slot";
      cell.title = "Free";
      if (range && range.status === "occupied" && range.mine) {
        cell.classList.add("mine");
        cell.title = `${range.title} (booking ${range.booking_id})`;
      } else if (range && range.status === "occupied") {
        cell.classList.add("occupied");
        cell.title = "Occupied";
      }
    }
  }
}

byId("day").addEventListener("change", loadSchedule);

state.token = recall("token");
state.username = recall("username");
if (state.token) api("/auth/me").then(showApp).catch(signOut);
</script>
</body>
</html>
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `75 passed`, lint clean.

- [ ] **Step 6: Check it in the browser**

Run: `uv run uvicorn app.main:create_app --factory --port 8000` (uses `.env` from Task 8).
Open `http://localhost:8000`, sign in as User1, click the first suggestion, then the second, confirm the card, and check that the grid shows room B 15:00–16:00 as yours. Sign in as User2 in a private window: the same cell shows as occupied without a title.

- [ ] **Step 7: Commit**

```bash
git add app/static/index.html app/main.py tests/test_api.py
git commit -m "Add the web page: login, chat, confirmation card and day schedule

There was no way to use the assistant without curl. One static page with no build step
shows the conversation, turns pending actions into Confirm/Cancel cards, and refreshes a
rooms-by-slots grid after every turn; assistant text is rendered as text, never as HTML."
```

---

### Task 11: Evals against the real model

**Files:**
- Create: `evals/__init__.py` (empty), `evals/test_evals.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `create_app`, `app.state.agent`, `app.state.service`, test support.
- Produces: `uv run pytest -m eval` — eleven cases over the five response paths.

- [ ] **Step 1: Register the evals folder**

In `pyproject.toml` change `testpaths = ["tests"]` to `testpaths = ["tests", "evals"]`.

- [ ] **Step 2: Write the evals**

`evals/test_evals.py`:
```python
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.support import FIXED_NOW, USER1_ID, USER2_ID, FakeClock, booking_request, login_headers

pytestmark = pytest.mark.eval


@pytest.fixture
def eval_app(tmp_path):
    settings = Settings()
    if settings.openai_api_key is None:
        pytest.skip("OPENAI_API_KEY is not configured")
    database_url = f"sqlite:///{(tmp_path / 'eval.db').as_posix()}"
    return create_app(settings.model_copy(update={"database_url": database_url}), clock=FakeClock(FIXED_NOW))


class Conversation:
    def __init__(self, app, username: str):
        self.app = app
        self.client = TestClient(app)
        self.headers = login_headers(self.client, username)
        self.user_id = self.client.get("/auth/me", headers=self.headers).json()["id"]
        self.conversation_id = None

    def say(self, message: str) -> dict:
        reply = self.client.post(
            "/chat/messages",
            json={"conversation_id": self.conversation_id, "message": message},
            headers=self.headers,
        ).json()
        self.conversation_id = reply["conversation_id"]
        return reply

    def decide(self, approve: bool = True) -> dict:
        return self.client.post(
            "/chat/decisions",
            json={"conversation_id": self.conversation_id, "approve": approve},
            headers=self.headers,
        ).json()

    def tool_calls(self, name: str) -> list[dict]:
        config = {"configurable": {"thread_id": f"{self.user_id}:{self.conversation_id}"}}
        calls = []
        for message in self.app.state.agent.get_state(config).values["messages"]:
            calls.extend(call for call in getattr(message, "tool_calls", []) if call["name"] == name)
        return calls


def test_answer_availability_with_the_availability_tool(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Which rooms are free tomorrow from 15:00 to 16:00 for 8 people?")
    [call] = user1.tool_calls("list_available_rooms")[:1]
    assert call["args"]["start"].startswith("2026-09-23T15:00")
    assert call["args"].get("attendees") == 8
    assert reply["pending_actions"] == []


def test_answer_a_schedule_without_other_peoples_titles(eval_app):
    eval_app.state.service.create(
        USER2_ID, booking_request(room_id="C", title="Secret merger talks", attendees=5)
    )
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("How does room C look tomorrow?")
    assert [call["args"]["room"].upper() for call in user1.tool_calls("get_room_schedule")][:1] == ["C"]
    assert "merger" not in reply["reply"].lower()


def test_clarify_instead_of_inventing_title_and_attendees(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Book me a room tomorrow at 10")
    assert reply["pending_actions"] == []
    assert user1.tool_calls("create_booking") == []


def test_clarify_the_number_of_attendees(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Book room B tomorrow from 10:00 to 11:00, title Standup")
    assert reply["pending_actions"] == []
    assert user1.tool_calls("create_booking") == []


def test_refuse_anything_outside_room_booking(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Write me a poem about the sea")
    assert reply["pending_actions"] == []
    assert reply["reply"]
    for name in ["list_available_rooms", "get_room_schedule", "list_my_bookings", "create_booking"]:
        assert user1.tool_calls(name) == []


def test_refuse_to_cancel_someone_elses_booking(eval_app):
    theirs = eval_app.state.service.create(
        USER2_ID, booking_request(room_id="D", title="Board prep", attendees=5)
    )
    user1 = Conversation(eval_app, "User1")
    reply = user1.say(f"Cancel booking {theirs['booking_id']}")
    assert reply["pending_actions"] == []
    assert eval_app.state.service.bookings_of(USER2_ID) == [theirs]


def test_structured_failure_when_the_room_is_too_small(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Book room A tomorrow from 10:00 to 11:00 for 30 people, title All hands")
    assert reply["pending_actions"] == []
    assert eval_app.state.service.bookings_of(USER1_ID) == []


def test_act_book_after_confirmation(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say('Book room B tomorrow from 10:00 to 11:30 for 4 people, title "Interview with John Doe"')
    [action] = reply["pending_actions"]
    assert action["tool"] == "create_booking"
    assert action["summary"].startswith("Book room B · Wed 23 Sep, 10:00–11:30")
    assert "Interview with John Doe" in action["summary"]
    assert action["summary"].endswith("4 attendees")
    user1.decide(approve=True)
    [booking] = eval_app.state.service.bookings_of(USER1_ID)
    assert (booking["room"], booking["start"], booking["end"]) == ("B", "2026-09-23T10:00", "2026-09-23T11:30")


def test_act_on_a_relative_date_in_spanish(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Reservame la sala E pasado mañana de 14 a 15 para 10 personas, título: Demo cliente")
    [action] = reply["pending_actions"]
    assert action["summary"].startswith("Book room E · Thu 24 Sep, 14:00–15:00")
    done = user1.decide(approve=True)
    assert "sala" in done["reply"].lower()
    assert [booking["room"] for booking in eval_app.state.service.bookings_of(USER1_ID)] == ["E"]


def test_act_on_a_correction_in_the_next_turn(eval_app):
    user1 = Conversation(eval_app, "User1")
    first = user1.say('Book room C tomorrow from 16:00 to 17:00 for 3 people, title "Sync"')
    assert first["pending_actions"][0]["summary"].endswith("3 attendees")
    second = user1.say("Actually make it 6 people")
    assert second["pending_actions"][0]["summary"].endswith("6 attendees")
    user1.decide(approve=True)
    [booking] = eval_app.state.service.bookings_of(USER1_ID)
    assert booking["attendees"] == 6


def test_act_cancel_my_own_booking(eval_app):
    eval_app.state.service.create(
        USER1_ID,
        booking_request(room_id="D", start="2026-09-23T12:00", end="2026-09-23T13:00", title="1:1", attendees=2),
    )
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Cancel my booking tomorrow at 12")
    assert [action["tool"] for action in reply["pending_actions"]] == ["cancel_booking"]
    user1.decide(approve=True)
    assert eval_app.state.service.bookings_of(USER1_ID) == []
```

- [ ] **Step 3: Check the default run still skips them**

Run: `uv run pytest -q`
Expected: `75 passed, 11 deselected`.

- [ ] **Step 4: Run the evals**

Run: `uv run pytest -m eval -v`
Expected: `11 passed`. When a case fails, print its thread (`user1.app.state.agent.get_state(...)`) and read the tool calls and tool results before changing anything. Fix the prompt or a tool description in `app/agent.py`, never the expectation, rerun the whole eval set, and keep the failure and its fix for the commit message.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add pyproject.toml evals app/agent.py
git commit -m "Add evals of the agent against the real model

Unit tests script the model, so nothing showed whether the real model takes the right path.
Eleven cases cover answer, clarify, refuse, structured failure and act, asserting on the
tools called and the final database state, not on wording."
```

---

### Task 12: Container, CI and deployment

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `railway.json`, `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `create_app` as a uvicorn factory, `/health`.
- Produces: a public URL (Railway domain) used by Task 13.

- [ ] **Step 1: Write the files**

`Dockerfile`:
```dockerfile
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev
COPY app ./app
CMD ["sh", "-c", "uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
```

`.dockerignore`:
```
.git
.venv
.env
*.db
__pycache__
.pytest_cache
.ruff_cache
tests
evals
doc
```

`railway.json`:
```json
{
  "$schema": "https://railway.com/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "healthcheckPath": "/health",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest
```

- [ ] **Step 2: Build and run the image locally**

```bash
docker build -t room-booking .
docker run --rm -d --name room-booking -p 8000:8000 -e JWT_SECRET=local -e OPENAI_API_KEY=sk-not-used room-booking
curl -s http://localhost:8000/health
docker stop room-booking
```
Expected: `{"status":"ok"}`.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile .dockerignore railway.json .github/workflows/ci.yml
git commit -m "Add Docker image, Railway config and CI

The app only ran on this machine. One image serves the API and the page on Railway's PORT
with a /health check; CI runs lint and the unit tests on every push, without evals."
```

- [ ] **Step 4: Publish the repository (ask the author first)**

The author creates the public repository `promtior-room-booking` on GitHub under their account (default assumed: `MatiasL0pez`, confirm) and says to push. Then:
```bash
git remote add origin https://github.com/MatiasL0pez/promtior-room-booking.git
git push -u origin main
```
Expected: the CI run on GitHub passes (Actions tab).

- [ ] **Step 5: Deploy on Railway (the author, in the Railway dashboard)**

1. New Project → Deploy from GitHub repo → `promtior-room-booking`.
2. In the project: New → Database → PostgreSQL.
3. Service → Variables: `DATABASE_URL=${{Postgres.DATABASE_URL}}`, `OPENAI_API_KEY` (the budgeted project key), `JWT_SECRET` (a new random value), `OPENAI_MODEL=gpt-6-luna`.
4. Service → Settings → Networking → Generate Domain.
5. Wait for the deployment: the healthcheck on `/health` must pass.

- [ ] **Step 6: Smoke test the public URL**

```bash
BASE=https://<the domain generated in step 5>
curl -s $BASE/health
TOKEN=$(curl -s -X POST $BASE/auth/login -d "username=User1&password=TechnicalChallengePromtior" | python -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
curl -s -X POST $BASE/chat/messages -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"message": "Which rooms are free tomorrow from 10 to 11?"}'
```
Expected: `{"status":"ok"}`, then a JSON reply listing free rooms. Then open `$BASE` in a browser and repeat the Task 10 Step 6 walkthrough.

---

### Task 13: Documentation — architecture, notebook, overview, README

**Files:**
- Create: `doc/architecture.md`, `doc/diagrams/components.png`, `doc/diagrams/sequence.png`, `doc/walkthrough.ipynb`, `doc/README.md`, `README.md`
- Modify: `pyproject.toml` / `uv.lock` (dev dependencies for the notebook)

**Interfaces:**
- Consumes: everything above; the public URL from Task 12 Step 5.

- [ ] **Step 1: Write `doc/architecture.md`**

````markdown
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
| `HumanInTheLoopMiddleware` | after each model call | pauses `create_booking` / `cancel_booking` only when the dry run passes |
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
````

- [ ] **Step 2: Export the diagrams to PNG**

```bash
npx -y @mermaid-js/mermaid-cli -i doc/architecture.md -o doc/diagrams/rendered.md -e png -b white
ls doc/diagrams
```
Expected: `rendered-1.png` and `rendered-2.png`. Then:
```bash
mv doc/diagrams/rendered-1.png doc/diagrams/components.png
mv doc/diagrams/rendered-2.png doc/diagrams/sequence.png
rm doc/diagrams/rendered.md
```
Open both PNGs and check that the text is readable.

- [ ] **Step 3: Build and execute the notebook**

Run: `uv add --dev nbformat nbconvert ipykernel`

Write this throwaway builder to the session scratchpad (not the repo) as `build_notebook.py`:
```python
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = [
    new_markdown_cell("""# Room Booking Assistant — technical walkthrough

Every section runs against the real modules in `app/`. Run it from the repository root with
`uv run jupyter nbconvert --execute --to notebook --inplace doc/walkthrough.ipynb`.
Section 6 talks to OpenAI and needs `OPENAI_API_KEY` in `.env`; the committed copy keeps its outputs.

| Technology | Role in the solution | Why this one |
|---|---|---|
| FastAPI | login, REST, chat API, static page | typed validation, OpenAPI at `/docs`, dependency injection for the current user |
| SQLAlchemy 2 · SQLite / Postgres | persistence | one model for local SQLite and Railway Postgres; the no-double-booking rule is a primary key |
| pwdlib (Argon2) · PyJWT | authentication | what the FastAPI documentation recommends today |
| LangChain `create_agent` on LangGraph | tool-calling agent | middleware for the confirmation gate, hidden runtime context for the user, checkpointed threads |
| OpenAI `gpt-6-luna`, Responses API | language understanding | the cheapest current model with function calling; switchable with `OPENAI_MODEL` |
| pytest | tests and evals | rules, API and agent wiring with a scripted model; evals with the real model |
| Docker · Railway · GitHub Actions | delivery | one image, managed Postgres, CI on every push |"""),
    new_markdown_cell("## 1. Setup: a throwaway database and a fixed clock\n\nThe clock is injected, so \"now\" is Tuesday 22 September 2026, 09:00 in Montevideo for every example."),
    new_code_cell("""import json
import os
import sys
import tempfile
from pathlib import Path

if Path.cwd().name == "doc":
    os.chdir(Path.cwd().parent)
sys.path.insert(0, str(Path.cwd()))

from app.booking import BookingError, BookingService
from app.config import Settings
from app.db import create_session_factory, seed
from tests.support import FIXED_NOW, FakeClock, booking_request

workspace = Path(tempfile.mkdtemp())
settings = Settings(jwt_secret="walkthrough", database_url=f"sqlite:///{(workspace / 'rules.db').as_posix()}")
session_factory = create_session_factory(settings.sqlalchemy_database_url)
seed(session_factory, settings.seed_password)
service = BookingService(session_factory, settings, FakeClock(FIXED_NOW))
print("Office time:", FIXED_NOW.isoformat())
print("Rooms:", service.rooms())"""),
    new_markdown_cell("## 2. The rules live in `BookingService`, not in the prompt\n\nEvery violation is a `BookingError` with a code and the data the assistant needs to explain it."),
    new_code_cell("""print("Created:", service.create(1, booking_request()))

attempts = {
    "30 people in room B": booking_request(attendees=30),
    "three and a half hours": booking_request(start="2026-09-23T13:00", end="2026-09-23T16:30"),
    "starts at 10:15": booking_request(start="2026-09-23T10:15", end="2026-09-23T11:00"),
    "overlaps the booking above": booking_request(start="2026-09-23T11:00", end="2026-09-23T12:00"),
    "yesterday": booking_request(start="2026-09-21T10:00", end="2026-09-21T11:00"),
    "after closing": booking_request(start="2026-09-23T19:30", end="2026-09-23T20:30"),
    "blank title": booking_request(start="2026-09-23T15:00", end="2026-09-23T16:00", title="  "),
}
for label, request in attempts.items():
    try:
        service.create(2, request)
    except BookingError as error:
        print(f"{label:28} -> {json.dumps(error.as_dict())}")"""),
    new_markdown_cell("## 3. No double booking, even under concurrency\n\nEach 30-minute slot is a row with primary key `(room_id, slot_start)`. Below, the second writer's check is skipped to simulate a request whose check ran before the first one committed: the database still refuses it."),
    new_code_cell("""service.check_create = lambda request: None
try:
    service.create(2, booking_request())
except BookingError as error:
    print("Second writer:", error.code, "-", error.message)
finally:
    del service.check_create"""),
    new_markdown_cell("## 4. Schedules and data minimization\n\nUser2 books room B too. User1 sees it only as occupied: the title written by User2 never reaches User1's LLM context."),
    new_code_cell("""service.create(2, booking_request(start="2026-09-23T14:00", end="2026-09-23T15:00", title="Secret merger talks"))
schedule = service.room_schedule(1, "B", service.parse_local_datetime("2026-09-23T00:00"), service.parse_local_datetime("2026-09-23T23:59"))
for entry in schedule["ranges"]:
    print(entry)"""),
    new_markdown_cell("## 5. Tools: what the model sees, and what it does not\n\nThe user reaches the tools through `ToolRuntime` context. It is not an argument, so it is not in the schema the model receives and no prompt can change it."),
    new_code_cell("""from app.agent import build_tools

for agent_tool in build_tools(service):
    properties = agent_tool.tool_call_schema.model_json_schema().get("properties", {})
    print(f"{agent_tool.name:22} {list(properties)}")"""),
    new_markdown_cell("""## 6. The agent end to end, with the confirmation gate

`create_agent` runs four middlewares: a dynamic system prompt, structured tool errors, `HumanInTheLoopMiddleware` (it pauses a write only when a dry run says it will succeed) and a limit of six model calls per message. This cell talks to the real model through the HTTP API."""),
    new_code_cell("""from fastapi.testclient import TestClient

from app.main import create_app
from tests.support import login_headers

agent_settings = settings.model_copy(update={"database_url": f"sqlite:///{(workspace / 'agent.db').as_posix()}"})
app = create_app(agent_settings, clock=FakeClock(FIXED_NOW))
client = TestClient(app)
headers = login_headers(client, "User1")


def say(message, conversation_id=None):
    reply = client.post("/chat/messages", json={"conversation_id": conversation_id, "message": message}, headers=headers).json()
    print("User1:     ", message)
    if reply["reply"]:
        print("Assistant: ", reply["reply"])
    for action in reply["pending_actions"]:
        print("Confirm?   ", action["summary"])
    return reply


first = say("Which rooms are free tomorrow from 10:00 to 11:00 for 6 people?")
second = say('Book room C then, title "Sprint planning"', first["conversation_id"])"""),
    new_code_cell("""done = client.post("/chat/decisions", json={"conversation_id": second["conversation_id"], "approve": True}, headers=headers).json()
print("Assistant: ", done["reply"])
print("Bookings of User1:", client.get("/bookings/mine", headers=headers).json())

failure = say("Book room A tomorrow from 12:00 to 13:00 for 10 people, title Workshop")"""),
    new_markdown_cell("## 7. The HTTP API\n\nThe same service is reachable without the LLM; the full OpenAPI documentation is at `/docs`."),
    new_code_cell("""for path, methods in app.openapi()["paths"].items():
    print(f"{', '.join(method.upper() for method in methods):10} {path}")"""),
    new_markdown_cell("## 8. Tests and evals\n\nUnit tests script the model and run in seconds without a key. Evals use the real model and assert on the tools called and the final database state, not on wording."),
    new_code_cell("""import subprocess

unit = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
print(unit.stdout.strip().splitlines()[-1])
evals = subprocess.run([sys.executable, "-m", "pytest", "-m", "eval", "-q"], capture_output=True, text=True)
print(evals.stdout.strip().splitlines()[-1])"""),
    new_markdown_cell("""## 9. Observability and deployment

- Logs: every refused tool call is logged with its error code; unexpected tool failures are logged with the traceback.
- Tracing: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to send every model call, tool call and interrupt to LangSmith without code changes.
- Deployment: one Docker image on Railway with managed Postgres; `/health` is the healthcheck; CI runs lint and unit tests on every push."""),
]

notebook = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
Path("doc/walkthrough.ipynb").write_text(nbformat.writes(notebook), encoding="utf-8")
print("written doc/walkthrough.ipynb")
```
Run from the repository root:
```bash
uv run python "<scratchpad>/build_notebook.py"
uv run jupyter nbconvert --execute --to notebook --inplace doc/walkthrough.ipynb
```
Expected: the notebook executes without errors. Open it and check every output: section 2 prints seven different error codes; section 3 prints `SLOT_TAKEN`; section 4 shows `"mine": False` without a title; section 5 shows no user field; section 6 shows a confirmation summary, then the booking in `/bookings/mine`, then a capacity explanation; section 8 prints `75 passed, 11 deselected` and `11 passed, 75 deselected`.

- [ ] **Step 4: Write `doc/README.md` (draft for the author to rewrite in their own words)**

```markdown
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
5. **Evals on behavior.** Unit tests script the model; evals run the real model and assert on the
   tools it called and on the final database state.

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
<!-- Add the challenges met while running the evals (Task 11 commit message). -->

## Assumptions

Room capacities A=4, B=6, C=8, D=12, E=20; bookable hours 08:00–20:00 on any day; a single time
zone, `America/Montevideo`. All three are configuration.

## Limitations and next steps

- Conversations live in memory: a redeploy forgets them (bookings persist). Next step: a Postgres
  checkpointer, which also allows more than one replica.
- Tables are created at startup; next step: Alembic migrations.
- No token streaming; the page shows a "Thinking…" state instead.
- Recurring bookings and per-user time zones are out of scope.
```
Replace the HTML comment with the real challenges from the Task 11 commit (or delete it if there were none).

- [ ] **Step 5: Write the root `README.md`**

```markdown
# Room Booking Assistant

[![CI](https://github.com/MatiasL0pez/promtior-room-booking/actions/workflows/ci.yml/badge.svg)](https://github.com/MatiasL0pez/promtior-room-booking/actions/workflows/ci.yml)

A chatbot that books the meeting rooms of the Cubo Itaú office through tool calling — the
Promtior AI Engineer technical challenge.

**Live demo:** <the Railway URL from Task 12> — sign in as `User1` or `User2`, password
`TechnicalChallengePromtior`. API documentation at `/docs`.

## What it does

- Checks which rooms are free for a time range, shows a room's schedule, lists your bookings.
- Books a room (title and attendees required) and cancels your own bookings, always after you
  confirm on a card.
- Enforces every rule in code and in the database: five rooms A–E with their capacity, 30-minute
  slots, at most 3 hours, no overlaps, business hours.

## How it works

FastAPI → LangChain `create_agent` (LangGraph runtime, OpenAI `gpt-6-luna`) → five tools →
`BookingService` → SQLite or Postgres. The user reaches the tools through runtime context, never
as an argument the model could forge; writes pause for confirmation only when a dry run says
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
uv run pytest            # unit tests: rules, API, agent wiring with a scripted model
uv run pytest -m eval    # evals with the real model (needs OPENAI_API_KEY)
```

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
```
Replace `<the Railway URL from Task 12>` with the real domain.

- [ ] **Step 6: Final checks and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
git add doc README.md pyproject.toml uv.lock
git commit -m "Add documentation: architecture, walkthrough notebook, overview and README

The challenge asks for an overview, a component diagram and a notebook that explains the
technologies with code from the solution. The notebook is committed executed so it reads
without an API key; the overview records the decisions and the challenges met."
```
Then the author pushes, checks CI and the rendered Mermaid on GitHub, and fills in the Microsoft form from the challenge PDF with the repository URL.
