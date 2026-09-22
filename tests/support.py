from datetime import datetime
from zoneinfo import ZoneInfo

from app.auth import CurrentUser
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


USER1 = CurrentUser(id=USER1_ID, username="User1")
USER2 = CurrentUser(id=USER2_ID, username="User2")


def login_headers(client, username: str) -> dict:
    response = client.post(
        "/auth/login", data={"username": username, "password": "TechnicalChallengePromtior"}
    )
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
