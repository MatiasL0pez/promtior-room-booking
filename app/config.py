from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    jwt_secret: str
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-6-luna"
    openai_reasoning_effort: str = "low"
    database_url: str = "sqlite:///./room_booking.db"
    seed_password: str = "TechnicalChallengePromtior"
    office_timezone: str = "America/Montevideo"
    opening_hour: int = 8
    closing_hour: int = 20
    token_lifetime_minutes: int = 480
    messages_per_window: int = 30
    message_window_minutes: int = 10

    @property
    def sqlalchemy_database_url(self) -> str:
        for prefix in ("postgres://", "postgresql://"):
            if self.database_url.startswith(prefix):
                return "postgresql+psycopg://" + self.database_url.removeprefix(prefix)
        return self.database_url
