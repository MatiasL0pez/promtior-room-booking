import pytest
from fastapi.testclient import TestClient
from lingua import Language, LanguageDetectorBuilder
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app
from tests.support import FIXED_NOW, USER1_ID, USER2_ID, FakeClock, booking_request, login_headers

pytestmark = pytest.mark.eval

LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.SPANISH
).build()


@pytest.fixture
def eval_app(tmp_path):
    try:
        settings = Settings()
    except ValidationError as error:
        pytest.skip(
            f"Settings are not configured: {error.error_count()} validation error(s), e.g. JWT_SECRET missing"
        )
    if settings.openai_api_key is None:
        pytest.skip("OPENAI_API_KEY is not configured")
    database_url = f"sqlite:///{(tmp_path / 'eval.db').as_posix()}"
    return create_app(
        settings.model_copy(update={"database_url": database_url}), clock=FakeClock(FIXED_NOW)
    )


class Conversation:
    def __init__(self, app, username: str):
        self.app = app
        self.client = TestClient(app)
        self.headers = login_headers(self.client, username)
        self.user_id = self.client.get("/auth/me", headers=self.headers).json()["id"]
        self.conversation_id = None

    def say(self, message: str) -> dict:
        reply = self.client.post(
            "/chat/messages",
            json={"conversation_id": self.conversation_id, "message": message},
            headers=self.headers,
        ).json()
        self.conversation_id = reply["conversation_id"]
        return reply

    def decide(self, approve: bool = True) -> dict:
        return self.client.post(
            "/chat/decisions",
            json={"conversation_id": self.conversation_id, "approve": approve},
            headers=self.headers,
        ).json()

    def tool_calls(self, name: str) -> list[dict]:
        config = {"configurable": {"thread_id": f"{self.user_id}:{self.conversation_id}"}}
        calls = []
        for message in self.app.state.agent.get_state(config).values["messages"]:
            calls.extend(
                call for call in getattr(message, "tool_calls", []) if call["name"] == name
            )
        return calls


def test_answer_availability_with_the_availability_tool(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Which rooms are free tomorrow from 15:00 to 16:00 for 8 people?")
    [call] = user1.tool_calls("list_available_rooms")[:1]
    assert call["args"]["start"].startswith("2026-09-23T15:00")
    assert call["args"].get("attendees") == 8
    assert reply["pending_actions"] == []


def test_answer_a_schedule_without_other_peoples_titles(eval_app):
    eval_app.state.service.create(
        USER2_ID, booking_request(room_id="C", title="Secret merger talks", attendees=5)
    )
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("How does room C look tomorrow?")
    assert [call["args"]["room"].upper() for call in user1.tool_calls("get_room_schedule")][:1] == [
        "C"
    ]
    assert "merger" not in reply["reply"].lower()


@pytest.mark.parametrize(
    ("message", "language"),
    [
        ("Which rooms are free tomorrow from 15:00 to 16:00 for 8 people?", Language.ENGLISH),
        ("¿Qué salas están libres mañana de 15 a 16 para 8 personas?", Language.SPANISH),
        (
            "Book room A tomorrow from 12:00 to 13:00 for 10 people, title Workshop",
            Language.ENGLISH,
        ),
    ],
    ids=["question_in_english", "question_in_spanish", "failed_booking_in_english"],
)
def test_reply_in_the_language_of_the_user(eval_app, message, language):
    reply = Conversation(eval_app, "User1").say(message)
    assert LANGUAGE_DETECTOR.detect_language_of(reply["reply"]) == language


def test_clarify_instead_of_inventing_title_and_attendees(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Book me a room tomorrow at 10")
    assert reply["pending_actions"] == []
    assert user1.tool_calls("create_booking") == []


def test_clarify_the_number_of_attendees(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Book room B tomorrow from 10:00 to 11:00, title Standup")
    assert reply["pending_actions"] == []
    assert user1.tool_calls("create_booking") == []


def test_list_my_bookings_on_screen_without_repeating_them(eval_app):
    for day in ("2026-09-23", "2026-09-24"):
        eval_app.state.service.create(
            USER1_ID,
            booking_request(
                room_id="C",
                start=f"{day}T15:00",
                end=f"{day}T17:00",
                title="Quarterly planning",
                attendees=6,
            ),
        )
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Show my bookings")
    assert len(reply["bookings"]) == 2
    assert "Quarterly planning" not in reply["reply"]


def test_refuse_anything_outside_room_booking(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Write me a poem about the sea")
    assert reply["pending_actions"] == []
    assert reply["reply"]
    for name in ["list_available_rooms", "get_room_schedule", "list_my_bookings", "create_booking"]:
        assert user1.tool_calls(name) == []


def test_refuse_to_cancel_someone_elses_booking(eval_app):
    theirs = eval_app.state.service.create(
        USER2_ID, booking_request(room_id="D", title="Board prep", attendees=5)
    )
    user1 = Conversation(eval_app, "User1")
    reply = user1.say(f"Cancel booking {theirs['booking_id']}")
    assert reply["pending_actions"] == []
    assert eval_app.state.service.bookings_of(USER2_ID) == [theirs]


def test_structured_failure_when_the_room_is_too_small(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Book room A tomorrow from 10:00 to 11:00 for 30 people, title All hands")
    assert reply["pending_actions"] == []
    assert eval_app.state.service.bookings_of(USER1_ID) == []


def test_offer_an_alternative_in_words_not_in_a_card(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Book room A tomorrow from 12:00 to 13:00 for 10 people, title Workshop")
    assert reply["pending_actions"] == []
    assert reply["reply"]
    assert user1.tool_calls("list_available_rooms") + user1.tool_calls("get_room_schedule")
    assert eval_app.state.service.bookings_of(USER1_ID) == []


def test_act_book_after_confirmation(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say(
        'Book room B tomorrow from 10:00 to 11:30 for 4 people, title "Interview with John Doe"'
    )
    [action] = reply["pending_actions"]
    assert action["tool"] == "create_booking"
    assert action["summary"].startswith("Book room B · Wed 23 Sep, 10:00–11:30")
    assert "Interview with John Doe" in action["summary"]
    assert action["summary"].endswith("4 attendees")
    user1.decide(approve=True)
    [booking] = eval_app.state.service.bookings_of(USER1_ID)
    assert (booking["room"], booking["start"], booking["end"]) == (
        "B",
        "2026-09-23T10:00",
        "2026-09-23T11:30",
    )


def test_act_on_a_relative_date_in_spanish(eval_app):
    user1 = Conversation(eval_app, "User1")
    reply = user1.say(
        "Reservame la sala E pasado mañana de 14 a 15 para 10 personas, título: Demo cliente"
    )
    [action] = reply["pending_actions"]
    assert action["summary"].startswith("Book room E · Thu 24 Sep, 14:00–15:00")
    user1.decide(approve=True)
    assert [booking["room"] for booking in eval_app.state.service.bookings_of(USER1_ID)] == ["E"]


def test_act_on_a_correction_in_the_next_turn(eval_app):
    user1 = Conversation(eval_app, "User1")
    first = user1.say('Book room C tomorrow from 16:00 to 17:00 for 3 people, title "Sync"')
    assert first["pending_actions"][0]["summary"].endswith("3 attendees")
    second = user1.say("Actually make it 6 people")
    assert second["pending_actions"][0]["summary"].endswith("6 attendees")
    user1.decide(approve=True)
    [booking] = eval_app.state.service.bookings_of(USER1_ID)
    assert booking["attendees"] == 6


def test_act_book_several_days_in_one_confirmation(eval_app):
    user1 = Conversation(eval_app, "User1")
    first = user1.say(
        "quiero reservar la sala d de 11:00 a 13:00 a partir de mañana y por 7 dias seguidos"
    )
    assert first["pending_actions"] == []
    assert user1.tool_calls("create_booking") == []
    second = user1.say("reunion cliente 8 personas")
    assert [action["tool"] for action in second["pending_actions"]] == ["create_booking"] * 7
    user1.decide(approve=True)
    bookings = eval_app.state.service.bookings_of(USER1_ID)
    assert sorted((booking["room"], booking["start"]) for booking in bookings) == [
        ("D", f"2026-09-{day}T11:00") for day in range(23, 30)
    ]


def test_act_cancel_my_own_booking(eval_app):
    eval_app.state.service.create(
        USER1_ID,
        booking_request(
            room_id="D", start="2026-09-23T12:00", end="2026-09-23T13:00", title="1:1", attendees=2
        ),
    )
    user1 = Conversation(eval_app, "User1")
    reply = user1.say("Cancel my booking tomorrow at 12")
    assert [action["tool"] for action in reply["pending_actions"]] == ["cancel_booking"]
    user1.decide(approve=True)
    assert eval_app.state.service.bookings_of(USER1_ID) == []
