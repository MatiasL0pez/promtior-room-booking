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
