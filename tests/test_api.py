from datetime import timedelta

import pytest

from app.auth import CurrentUser, create_access_token
from tests.support import USER1_ID, USER2_ID, booking_request


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
    [
        ("test-secret-long-enough-for-hs256-signing", timedelta(minutes=-1)),
        ("another-secret-long-enough-for-hs256-signing", timedelta(minutes=5)),
    ],
)
def test_expired_or_foreign_tokens_are_rejected(client, secret, lifetime):
    token = create_access_token(CurrentUser(id=1, username="User1"), secret, lifetime)
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


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
        {
            "start": "2026-09-23T10:00",
            "end": "2026-09-23T11:30",
            "status": "occupied",
            "mine": False,
        }
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
