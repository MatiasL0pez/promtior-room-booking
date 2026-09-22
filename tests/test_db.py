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
