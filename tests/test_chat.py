from datetime import timedelta
from threading import Barrier, Thread

import httpx
import openai
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from app.agent import build_agent
from app.chat import Conversations, NothingToDecide
from app.main import create_app
from tests.support import USER1, login_headers, scripted, tool_call


class UnavailableChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise openai.APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com/v1/responses")
        )


CREATE_B = {
    "room": "B",
    "start": "2026-09-23T10:00",
    "end": "2026-09-23T11:30",
    "title": "Interview",
    "attendees": 4,
}


def send(client, headers, message, conversation_id=None):
    return client.post(
        "/chat/messages",
        json={"conversation_id": conversation_id, "message": message},
        headers=headers,
    )


def decide(client, headers, conversation_id, approve=True):
    return client.post(
        "/chat/decisions",
        json={"conversation_id": conversation_id, "approve": approve},
        headers=headers,
    )


def test_a_booking_through_chat_waits_for_confirmation(make_client):
    client = make_client(tool_call("create_booking", **CREATE_B), AIMessage(content="Booked."))
    headers = login_headers(client, "User1")
    paused = send(client, headers, "book B tomorrow 10 to 11:30").json()
    assert paused["pending_actions"] == [
        {
            "tool": "create_booking",
            "summary": "Book room B · Wed 23 Sep, 10:00–11:30 · Interview · 4 attendees",
        }
    ]
    assert client.get("/bookings/mine", headers=headers).json() == []

    done = decide(client, headers, paused["conversation_id"]).json()
    assert done == {
        "conversation_id": paused["conversation_id"],
        "reply": "Booked.",
        "pending_actions": [],
    }
    assert [
        booking["room"] for booking in client.get("/bookings/mine", headers=headers).json()
    ] == ["B"]


def test_confirming_twice_books_once(make_client):
    client = make_client(tool_call("create_booking", **CREATE_B), AIMessage(content="Booked."))
    headers = login_headers(client, "User1")
    paused = send(client, headers, "book B").json()
    assert decide(client, headers, paused["conversation_id"]).status_code == 200
    assert decide(client, headers, paused["conversation_id"]).status_code == 409
    assert len(client.get("/bookings/mine", headers=headers).json()) == 1


def test_typing_while_a_confirmation_is_pending_rejects_it_with_that_message(make_client):
    client = make_client(
        tool_call("create_booking", **CREATE_B),
        tool_call("create_booking", **{**CREATE_B, "attendees": 6}),
        AIMessage(content="Booked for 6."),
    )
    headers = login_headers(client, "User1")
    first = send(client, headers, "book B for 4").json()
    second = send(client, headers, "make it 6 people", first["conversation_id"]).json()
    assert second["pending_actions"][0]["summary"].endswith("6 attendees")

    state = client.app.state.agent.get_state(
        {"configurable": {"thread_id": f"1:{first['conversation_id']}"}}
    )
    rejections = [m.content for m in state.values["messages"] if isinstance(m, ToolMessage)]
    assert "make it 6 people" in rejections[0]

    decide(client, headers, first["conversation_id"])
    [booking] = client.get("/bookings/mine", headers=headers).json()
    assert booking["attendees"] == 6


def test_conversations_are_private_to_their_user(make_client):
    client = make_client(tool_call("create_booking", **CREATE_B))
    user1 = login_headers(client, "User1")
    user2 = login_headers(client, "User2")
    paused = send(client, user1, "book B").json()
    assert decide(client, user2, paused["conversation_id"]).status_code == 409
    assert client.get("/bookings/mine", headers=user1).json() == []


def test_a_decision_without_a_pending_action_is_a_conflict(client, user1_headers):
    assert decide(client, user1_headers, "nothing-here").status_code == 409


def test_long_messages_are_rejected(client, user1_headers):
    assert send(client, user1_headers, "x" * 1001).status_code == 422


def test_empty_conversation_ids_are_rejected(client, user1_headers):
    assert send(client, user1_headers, "hi", conversation_id="").status_code == 422
    assert decide(client, user1_headers, "").status_code == 422


def test_concurrent_decisions_on_the_same_conversation_serialize(service):
    agent = build_agent(
        service,
        scripted(
            tool_call("create_booking", **CREATE_B),
            AIMessage(content="Booked."),
            AIMessage(content="Booked again."),
        ),
    )
    conversations = Conversations(agent)
    conversations.send_message(USER1, "c1", "book B")

    barrier = Barrier(2)
    outcomes = []

    def decide():
        barrier.wait()
        try:
            outcomes.append(conversations.decide(USER1, "c1", True))
        except NothingToDecide as error:
            outcomes.append(error)

    threads = [Thread(target=decide) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    replies = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, NothingToDecide)]
    assert len(replies) == 1
    assert len(conflicts) == 1
    assert len(service.bookings_of(USER1.id)) == 1


def test_model_outages_answer_502(settings, clock):
    client = TestClient(
        create_app(settings, chat_model=UnavailableChatModel(messages=iter([])), clock=clock)
    )
    headers = login_headers(client, "User1")
    response = send(client, headers, "book B")
    assert response.status_code == 502
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_messages_are_rate_limited_per_user(settings, clock):
    limited = settings.model_copy(update={"messages_per_window": 2})
    model = scripted(AIMessage(content="a"), AIMessage(content="b"), AIMessage(content="c"))
    client = TestClient(create_app(limited, chat_model=model, clock=clock))
    headers = login_headers(client, "User1")
    statuses = [send(client, headers, "hi").status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    clock.now += timedelta(minutes=10)
    assert send(client, headers, "hi").status_code == 200
