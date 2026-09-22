from pwdlib import PasswordHash
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base, Room, User

ROOM_CAPACITIES = {"A": 4, "B": 6, "C": 8, "D": 12, "E": 20}
USERNAMES = ["User1", "User2"]

password_hasher = PasswordHash.recommended()


def create_session_factory(database_url: str) -> sessionmaker[Session]:
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def seed(session_factory: sessionmaker[Session], password: str) -> None:
    with session_factory.begin() as session:
        for room_id, capacity in ROOM_CAPACITIES.items():
            if session.get(Room, room_id) is None:
                session.add(Room(id=room_id, capacity=capacity))
        for username in USERNAMES:
            if session.scalar(select(User).where(User.username == username)) is None:
                session.add(User(username=username, password_hash=password_hasher.hash(password)))
