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
    morning = service.create(
        USER1_ID, booking_request(start="2026-09-23T08:00", end="2026-09-23T11:00")
    )
    evening = service.create(
        USER1_ID, booking_request(start="2026-09-23T17:00", end="2026-09-23T20:00")
    )
    assert (morning["start"], morning["end"]) == ("2026-09-23T08:00", "2026-09-23T11:00")
    assert (evening["start"], evening["end"]) == ("2026-09-23T17:00", "2026-09-23T20:00")


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
    assert error.value.details == {
        "conflict": {"start": "2026-09-23T10:00", "end": "2026-09-23T11:30"}
    }


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
