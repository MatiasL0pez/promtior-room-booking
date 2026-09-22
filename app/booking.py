from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models import Booking, BookingSlot, Room

SLOT_LENGTH = timedelta(minutes=30)
MAX_DURATION = timedelta(hours=3)
MAX_QUERY_RANGE = timedelta(days=7)
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
                if (
                    local_value.minute not in (0, 30)
                    or local_value.second
                    or local_value.microsecond
                ):
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

    def check_cancel(self, user_id: int, booking_id: int) -> dict:
        with self.session_factory() as session:
            booking = session.get(Booking, booking_id)
            if booking is None:
                raise BookingError("BOOKING_NOT_FOUND", f"There is no booking {booking_id}.")
            if booking.user_id != user_id:
                raise BookingError("NOT_OWNER", "Only the person who made a booking can cancel it.")
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
