import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from app.agent import build_agent, build_openai_model
from app.auth import CurrentUser, authenticate, create_access_token, read_access_token
from app.booking import BookingError, BookingService
from app.chat import Conversations, MessageRateLimiter, NothingToDecide
from app.config import Settings
from app.db import create_session_factory, seed


class MessageIn(BaseModel):
    conversation_id: str | None = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=1000)


class DecisionIn(BaseModel):
    conversation_id: str = Field(max_length=64)
    approve: bool


def system_clock() -> datetime:
    return datetime.now(UTC)


def create_app(settings: Settings | None = None, chat_model=None, clock=None) -> FastAPI:
    if settings is None:
        settings = Settings()
    if clock is None:
        clock = system_clock
    if chat_model is None:
        chat_model = build_openai_model(settings)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    session_factory = create_session_factory(settings.sqlalchemy_database_url)
    seed(session_factory, settings.seed_password)
    service = BookingService(session_factory, settings, clock)
    agent = build_agent(service, chat_model)
    conversations = Conversations(agent)
    rate_limiter = MessageRateLimiter(
        settings.messages_per_window, timedelta(minutes=settings.message_window_minutes), clock
    )
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

    app = FastAPI(title="Room Booking Assistant")
    app.state.service = service
    app.state.agent = agent

    def current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> CurrentUser:
        user = read_access_token(token, settings.jwt_secret)
        if user is None:
            raise HTTPException(
                401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
            )
        return user

    def within_rate_limit(user: Annotated[CurrentUser, Depends(current_user)]) -> CurrentUser:
        if not rate_limiter.allow(user.id):
            raise HTTPException(429, "Too many messages. Try again in a few minutes.")
        return user

    @app.exception_handler(BookingError)
    def booking_error_response(request, error: BookingError) -> JSONResponse:
        if error.code.endswith("NOT_FOUND"):
            return JSONResponse(error.as_dict(), status_code=404)
        return JSONResponse(error.as_dict(), status_code=400)

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

    @app.post("/chat/messages")
    def send_message(
        body: MessageIn, user: Annotated[CurrentUser, Depends(within_rate_limit)]
    ) -> dict:
        return conversations.send_message(user, body.conversation_id, body.message)

    @app.post("/chat/decisions")
    def decide(body: DecisionIn, user: Annotated[CurrentUser, Depends(within_rate_limit)]) -> dict:
        try:
            return conversations.decide(user, body.conversation_id, body.approve)
        except NothingToDecide:
            raise HTTPException(409, "There is no action waiting for confirmation.") from None

    return app
