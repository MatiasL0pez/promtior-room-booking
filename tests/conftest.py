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
