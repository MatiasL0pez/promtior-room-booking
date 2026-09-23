import json
import logging
from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from app.auth import CurrentUser
from app.booking import BookingError, BookingRequest, BookingService
from app.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the meeting room assistant of the Promtior office at Cubo Itaú, Montevideo. Your only job is to help {username} check room availability, book rooms and cancel their own bookings.

Now it is {now} ({weekday}) in {timezone}. Resolve relative dates such as "tomorrow" or "next Friday" from this moment. Pass times to tools as local ISO 8601 without offset, for example 2026-09-23T10:00.

Rooms and capacities: {rooms}.

Rules the booking system enforces:
- Bookings start and end on the hour or the half hour, last at most 3 hours, and fit between {opening} and {closing} on a single day.
- The attendees must fit in the room. Every booking needs a title.
- A room holds one booking at a time.

How to work:
- To book you need the room, the date, the start time, the end time or duration, the title and the number of attendees. Ask for whatever is missing. Never make up a title or a number of attendees.
- If the user has no room preference, check availability and suggest the smallest free room that fits.
- Do not ask for confirmation in text before creating or cancelling: the system shows the user a confirmation card by itself.
- When a tool answers with "ok": false, explain the reason in plain words and offer the closest alternative, such as another time or another room.
- To cancel, find the booking id with list_my_bookings. Never guess an id. Users can only cancel their own bookings.
- Other people's bookings appear only as occupied. Do not speculate about them.
- Politely decline anything that is not about meeting room bookings.
- Answer in the language the user writes in, briefly, in plain text without Markdown."""


def build_openai_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        use_responses_api=True,
        reasoning={"effort": settings.openai_reasoning_effort},
        timeout=30,
        max_retries=2,
        model_kwargs={"parallel_tool_calls": False},
    )


def booking_request_from(service: BookingService, arguments: dict) -> BookingRequest:
    return BookingRequest(
        room_id=arguments["room"].strip().upper(),
        start=service.parse_local_datetime(arguments["start"]),
        end=service.parse_local_datetime(arguments["end"]),
        title=arguments["title"],
        attendees=arguments["attendees"],
    )


def validated_arguments(agent_tool, raw_arguments: dict) -> dict:
    return agent_tool.tool_call_schema.model_validate(raw_arguments).model_dump()


def describe_time_range(service: BookingService, start: datetime, end: datetime) -> str:
    local_start = start.astimezone(service.timezone)
    local_end = end.astimezone(service.timezone)
    return (
        f"{local_start:%a} {local_start.day} {local_start:%b}, "
        f"{local_start:%H:%M}–{local_end:%H:%M}"
    )


@wrap_tool_call
def booking_errors_as_tool_results(request, handler):
    thread_id = request.runtime.config.get("configurable", {}).get("thread_id")
    name = request.tool_call["name"]
    try:
        result = handler(request)
        logger.info("thread=%s tool=%s ok", thread_id, name)
        return result
    except BookingError as error:
        logger.info("thread=%s Tool %s refused: %s", thread_id, name, error.code)
        result = error.as_dict()
    except Exception:
        logger.exception("thread=%s Tool %s failed", thread_id, name)
        result = {
            "ok": False,
            "error": "INTERNAL_ERROR",
            "message": "The booking system failed. Ask the user to try again in a moment.",
        }
    return ToolMessage(
        content=json.dumps(result),
        tool_call_id=request.tool_call["id"],
        name=name,
        status="error",
    )


def build_tools(service: BookingService) -> list:
    @tool(parse_docstring=True)
    def list_available_rooms(start: str, end: str, attendees: int | None = None) -> dict:
        """List the rooms that are free for the whole time range.

        Args:
            start: Local start time, ISO 8601 without offset, for example 2026-09-23T10:00.
            end: Local end time, ISO 8601 without offset.
            attendees: Keep only rooms with at least this capacity.
        """
        rooms = service.available_rooms(
            service.parse_local_datetime(start), service.parse_local_datetime(end), attendees
        )
        return {"ok": True, "rooms": rooms}

    @tool(parse_docstring=True)
    def get_room_schedule(
        room: str, start: str, end: str, runtime: ToolRuntime[CurrentUser]
    ) -> dict:
        """Show which parts of a time range are free or occupied in one room.

        Args:
            room: Room letter, A to E.
            start: Local start time, ISO 8601 without offset.
            end: Local end time, ISO 8601 without offset.
        """
        schedule = service.room_schedule(
            runtime.context.id,
            room.strip().upper(),
            service.parse_local_datetime(start),
            service.parse_local_datetime(end),
        )
        return {"ok": True, **schedule}

    @tool
    def list_my_bookings(runtime: ToolRuntime[CurrentUser]) -> dict:
        """List the current user's upcoming bookings with their ids."""
        return {"ok": True, "bookings": service.bookings_of(runtime.context.id)}

    @tool(parse_docstring=True)
    def create_booking(
        room: str,
        start: str,
        end: str,
        title: str,
        attendees: int,
        runtime: ToolRuntime[CurrentUser],
    ) -> dict:
        """Book a room for the current user. The system asks the user to confirm before it runs.

        Args:
            room: Room letter, A to E.
            start: Local start time, ISO 8601 without offset, on the hour or half hour.
            end: Local end time, ISO 8601 without offset, on the hour or half hour.
            title: Title of the meeting exactly as the user said it. Never invent one; if the user gave none, ask instead of calling this tool.
            attendees: Number of people attending exactly as the user said it. Never assume one; if the user gave none, ask instead of calling this tool.
        """
        arguments = {
            "room": room,
            "start": start,
            "end": end,
            "title": title,
            "attendees": attendees,
        }
        booking = service.create(runtime.context.id, booking_request_from(service, arguments))
        return {"ok": True, "booking": booking}

    @tool(parse_docstring=True)
    def cancel_booking(booking_id: int, runtime: ToolRuntime[CurrentUser]) -> dict:
        """Cancel one of the current user's bookings. The system asks the user to confirm before it runs.

        Args:
            booking_id: Id of the booking, from list_my_bookings.
        """
        return {"ok": True, "cancelled": service.cancel(runtime.context.id, booking_id)}

    return [
        list_available_rooms,
        get_room_schedule,
        list_my_bookings,
        create_booking,
        cancel_booking,
    ]


def build_agent(service: BookingService, chat_model):
    tools = build_tools(service)
    tools_by_name = {agent_tool.name: agent_tool for agent_tool in tools}
    rooms_text = ", ".join(
        f"{room['room']} ({room['capacity']} people)" for room in service.rooms()
    )

    @dynamic_prompt
    def system_prompt(request: ModelRequest) -> str:
        now = service.clock().astimezone(service.timezone)
        return SYSTEM_PROMPT.format(
            username=request.runtime.context.username,
            now=now.strftime("%Y-%m-%dT%H:%M"),
            weekday=now.strftime("%A"),
            timezone=service.timezone.key,
            rooms=rooms_text,
            opening=f"{service.opening_hour:02d}:00",
            closing=f"{service.closing_hour:02d}:00",
        )

    def create_would_succeed(request) -> bool:
        try:
            arguments = validated_arguments(
                tools_by_name["create_booking"], request.tool_call["args"]
            )
            service.check_create(booking_request_from(service, arguments))
        except (BookingError, ValidationError):
            return False
        return True

    def cancel_would_succeed(request) -> bool:
        try:
            arguments = validated_arguments(
                tools_by_name["cancel_booking"], request.tool_call["args"]
            )
            service.check_cancel(request.runtime.context.id, arguments["booking_id"])
        except (BookingError, ValidationError):
            return False
        return True

    def describe_create(tool_call, state, runtime) -> str:
        arguments = validated_arguments(tools_by_name["create_booking"], tool_call["args"])
        request = booking_request_from(service, arguments)
        attendees = f"{request.attendees} attendees"
        if request.attendees == 1:
            attendees = "1 attendee"
        time_range = describe_time_range(service, request.start, request.end)
        return f"Book room {request.room_id} · {time_range} · {request.title.strip()} · {attendees}"

    def describe_cancel(tool_call, state, runtime) -> str:
        arguments = validated_arguments(tools_by_name["cancel_booking"], tool_call["args"])
        booking = service.check_cancel(runtime.context.id, arguments["booking_id"])
        time_range = describe_time_range(
            service,
            service.parse_local_datetime(booking["start"]),
            service.parse_local_datetime(booking["end"]),
        )
        return (
            f"Cancel booking {booking['booking_id']} · room {booking['room']} · "
            f"{time_range} · {booking['title']}"
        )

    confirmation_gate = HumanInTheLoopMiddleware(
        interrupt_on={
            "create_booking": {
                "allowed_decisions": ["approve", "reject"],
                "when": create_would_succeed,
                "description": describe_create,
            },
            "cancel_booking": {
                "allowed_decisions": ["approve", "reject"],
                "when": cancel_would_succeed,
                "description": describe_cancel,
            },
        }
    )
    return create_agent(
        chat_model,
        tools=tools,
        middleware=[
            system_prompt,
            booking_errors_as_tool_results,
            confirmation_gate,
            ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),
        ],
        context_schema=CurrentUser,
        checkpointer=InMemorySaver(),
    )
