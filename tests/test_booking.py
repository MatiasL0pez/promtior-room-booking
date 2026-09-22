import pytest
from sqlalchemy import select

from app.booking import BookingError
from app.models import Booking, BookingSlot
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


def test_available_rooms_raises_invalid_time_range_when_end_equals_start(service):
    with pytest.raises(BookingError) as error:
        service.available_rooms(local("2026-09-23T10:00"), local("2026-09-23T10:00"))
    assert error.value.code == "INVALID_TIME_RANGE"


def test_schedule_hides_other_peoples_titles(service):
    service.create(USER1_ID, booking_request(title="Interview"))
    service.create(
        USER2_ID,
        booking_request(
            start="2026-09-23T14:00", end="2026-09-23T15:00", title="Secret merger talks"
        ),
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
            {
                "start": "2026-09-23T14:00",
                "end": "2026-09-23T15:00",
                "status": "occupied",
                "mine": False,
            },
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


def test_schedule_raises_invalid_time_range_when_end_equals_start(service):
    with pytest.raises(BookingError) as error:
        service.room_schedule(USER1_ID, "B", local("2026-09-23T10:00"), local("2026-09-23T10:00"))
    assert error.value.code == "INVALID_TIME_RANGE"


def test_bookings_of_lists_only_own_active_future_bookings(service, clock):
    service.create(
        USER1_ID, booking_request(start="2026-09-23T15:00", end="2026-09-23T16:00", title="Later")
    )
    service.create(USER1_ID, booking_request(title="Earlier"))
    dropped = service.create(USER1_ID, booking_request(room_id="C", title="Dropped", attendees=2))
    service.cancel(USER1_ID, dropped["booking_id"])
    service.create(USER2_ID, booking_request(room_id="D", title="Not mine"))
    assert [booking["title"] for booking in service.bookings_of(USER1_ID)] == ["Earlier", "Later"]
    clock.now = local("2026-09-23T12:00")
    assert [booking["title"] for booking in service.bookings_of(USER1_ID)] == ["Later"]
