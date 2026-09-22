import json

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from app.agent import build_agent, build_tools
from tests.support import USER1, USER2_ID, booking_request, scripted, tool_call

APPROVE = Command(resume={"decisions": [{"type": "approve"}]})
REJECT = Command(resume={"decisions": [{"type": "reject"}]})
CREATE_B = {
    "room": "b",
    "start": "2026-09-23T10:00",
    "end": "2026-09-23T11:30",
    "title": "Interview",
    "attendees": 4,
}


def run(agent, agent_input):
    return agent.invoke(
        agent_input, {"configurable": {"thread_id": "test"}}, context=USER1, version="v2"
    )


def ask(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def tool_results(result) -> list[dict]:
    return [
        json.loads(message.content)
        for message in result.value["messages"]
        if isinstance(message, ToolMessage)
    ]


def test_tools_never_expose_the_user_to_the_model(service):
    for agent_tool in build_tools(service):
        properties = agent_tool.tool_call_schema.model_json_schema().get("properties", {})
        assert "runtime" not in properties
        assert [name for name in properties if "user" in name] == []


def test_valid_booking_pauses_and_approval_writes_it(service):
    agent = build_agent(
        service, scripted(tool_call("create_booking", **CREATE_B), AIMessage(content="Booked."))
    )
    paused = run(agent, ask("book B"))
    [action] = paused.interrupts[0].value["action_requests"]
    assert (
        action["description"] == "Book room B · Wed 23 Sep, 10:00–11:30 · Interview · 4 attendees"
    )
    assert service.bookings_of(USER1.id) == []

    finished = run(agent, APPROVE)
    assert finished.interrupts == ()
    assert [booking["room"] for booking in service.bookings_of(USER1.id)] == ["B"]
    assert finished.value["messages"][-1].text == "Booked."


def test_invalid_booking_does_not_pause_and_writes_nothing(service):
    too_many = {**CREATE_B, "room": "A", "attendees": 30}
    agent = build_agent(
        service, scripted(tool_call("create_booking", **too_many), AIMessage(content="Too small."))
    )
    result = run(agent, ask("book A for 30"))
    assert result.interrupts == ()
    assert tool_results(result)[0]["error"] == "CAPACITY_EXCEEDED"
    assert service.bookings_of(USER1.id) == []


def test_rejection_writes_nothing(service):
    agent = build_agent(
        service, scripted(tool_call("create_booking", **CREATE_B), AIMessage(content="Ok."))
    )
    run(agent, ask("book B"))
    run(agent, REJECT)
    assert service.bookings_of(USER1.id) == []


def test_cancelling_someone_elses_booking_does_not_pause(service):
    theirs = service.create(USER2_ID, booking_request())
    agent = build_agent(
        service,
        scripted(
            tool_call("cancel_booking", booking_id=theirs["booking_id"]),
            AIMessage(content="Not yours."),
        ),
    )
    result = run(agent, ask(f"cancel booking {theirs['booking_id']}"))
    assert result.interrupts == ()
    assert tool_results(result)[0]["error"] == "NOT_OWNER"
    assert service.bookings_of(USER2_ID) == [theirs]


def test_cancelling_my_booking_pauses_with_its_summary(service):
    mine = service.create(USER1.id, booking_request())
    agent = build_agent(
        service,
        scripted(
            tool_call("cancel_booking", booking_id=mine["booking_id"]),
            AIMessage(content="Cancelled."),
        ),
    )
    paused = run(agent, ask("cancel it"))
    [action] = paused.interrupts[0].value["action_requests"]
    assert action["description"] == (
        "Cancel booking 1 · room B · Wed 23 Sep, 10:00–11:30 · Interview with John Doe"
    )
    run(agent, APPROVE)
    assert service.bookings_of(USER1.id) == []


def test_system_prompt_carries_the_user_and_the_office_time(service):
    model = scripted(AIMessage(content="Hi!"))
    run(build_agent(service, model), ask("hello"))
    system_message = model.received[0][0]
    assert system_message.type == "system"
    assert "User1" in system_message.text
    assert "2026-09-22T09:00 (Tuesday)" in system_message.text
    assert "A (4 people)" in system_message.text
