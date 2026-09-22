from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db import password_hasher
from app.models import User

# Verifying against a throwaway hash keeps the response time equal for unknown usernames.
HASH_FOR_UNKNOWN_USERS = password_hasher.hash("unknown-user")


@dataclass(frozen=True)
class CurrentUser:
    id: int
    username: str


def authenticate(
    session_factory: sessionmaker[Session], username: str, password: str
) -> CurrentUser | None:
    with session_factory() as session:
        user = session.scalar(select(User).where(func.lower(User.username) == username.lower()))
    if user is None:
        password_hasher.verify(password, HASH_FOR_UNKNOWN_USERS)
        return None
    if not password_hasher.verify(password, user.password_hash):
        return None
    return CurrentUser(id=user.id, username=user.username)


def create_access_token(user: CurrentUser, secret: str, lifetime: timedelta) -> str:
    claims = {"sub": str(user.id), "username": user.username, "exp": datetime.now(UTC) + lifetime}
    return jwt.encode(claims, secret, algorithm="HS256")


def read_access_token(token: str, secret: str) -> CurrentUser | None:
    try:
        claims = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        return None
    return CurrentUser(id=int(claims["sub"]), username=claims["username"])
