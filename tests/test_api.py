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
