import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.auth import CurrentUser, authenticate, create_access_token, read_access_token
from app.booking import BookingError, BookingService
from app.config import Settings
from app.db import create_session_factory, seed


def system_clock() -> datetime:
    return datetime.now(UTC)


def create_app(settings: Settings | None = None, clock=None) -> FastAPI:
    if settings is None:
        settings = Settings()
    if clock is None:
        clock = system_clock
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    session_factory = create_session_factory(settings.sqlalchemy_database_url)
    seed(session_factory, settings.seed_password)
    service = BookingService(session_factory, settings, clock)
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

    app = FastAPI(title="Room Booking Assistant")
    app.state.service = service

    def current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> CurrentUser:
        user = read_access_token(token, settings.jwt_secret)
        if user is None:
            raise HTTPException(
                401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
            )
        return user

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/auth/login")
    def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> dict:
        user = authenticate(session_factory, form.username, form.password)
        if user is None:
            raise HTTPException(
                401, "Wrong username or password", headers={"WWW-Authenticate": "Bearer"}
            )
        lifetime = timedelta(minutes=settings.token_lifetime_minutes)
        return {
            "access_token": create_access_token(user, settings.jwt_secret, lifetime),
            "token_type": "bearer",
            "username": user.username,
        }

    @app.get("/auth/me")
    def me(user: Annotated[CurrentUser, Depends(current_user)]) -> dict:
        return {"id": user.id, "username": user.username}

    @app.exception_handler(BookingError)
    def booking_error_response(request, error: BookingError) -> JSONResponse:
        if error.code.endswith("NOT_FOUND"):
            return JSONResponse(error.as_dict(), status_code=404)
        return JSONResponse(error.as_dict(), status_code=400)

    @app.get("/rooms")
    def list_rooms(user: Annotated[CurrentUser, Depends(current_user)]) -> list[dict]:
        return service.rooms()

    @app.get("/rooms/{room_id}/schedule")
    def room_schedule(
        room_id: str, start: str, end: str, user: Annotated[CurrentUser, Depends(current_user)]
    ) -> dict:
        return service.room_schedule(
            user.id,
            room_id.upper(),
            service.parse_local_datetime(start),
            service.parse_local_datetime(end),
        )

    @app.get("/bookings/mine")
    def my_bookings(user: Annotated[CurrentUser, Depends(current_user)]) -> list[dict]:
        return service.bookings_of(user.id)

    return app
