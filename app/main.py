from fastapi import FastAPI

from app.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    app = FastAPI(title="Room Booking Assistant")
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
