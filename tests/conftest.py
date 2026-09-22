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
