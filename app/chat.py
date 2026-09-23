import logging
from collections import defaultdict, deque
from datetime import timedelta
from threading import Lock
from uuid import uuid4

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.auth import CurrentUser

logger = logging.getLogger(__name__)


class NothingToDecide(Exception):
    pass


def thread_config(user: CurrentUser, conversation_id: str) -> dict:
    return {"configurable": {"thread_id": f"{user.id}:{conversation_id}"}}


class Conversations:
    def __init__(self, agent):
        self.agent = agent
        self.thread_locks = defaultdict(Lock)
        self.thread_locks_guard = Lock()

    def lock_of(self, thread_id: str) -> Lock:
        with self.thread_locks_guard:
            return self.thread_locks[thread_id]

    def send_message(self, user: CurrentUser, conversation_id: str | None, message: str) -> dict:
        if conversation_id is None:
            conversation_id = uuid4().hex
        config = thread_config(user, conversation_id)
        with self.lock_of(config["configurable"]["thread_id"]):
            pending = self.agent.get_state(config).interrupts
            if pending:
                action_count = len(pending[0].value["action_requests"])
                rejection = {"type": "reject", "message": message}
                agent_input = Command(resume={"decisions": [rejection] * action_count})
            else:
                agent_input = {"messages": [{"role": "user", "content": message}]}
            return self.answer(user, conversation_id, agent_input)

    def decide(self, user: CurrentUser, conversation_id: str, approve: bool) -> dict:
        config = thread_config(user, conversation_id)
        with self.lock_of(config["configurable"]["thread_id"]):
            pending = self.agent.get_state(config).interrupts
            if not pending:
                raise NothingToDecide()
            decision = {"type": "reject"}
            if approve:
                decision = {"type": "approve"}
            action_count = len(pending[0].value["action_requests"])
            agent_input = Command(resume={"decisions": [decision] * action_count})
            return self.answer(user, conversation_id, agent_input)

    def answer(self, user: CurrentUser, conversation_id: str, agent_input) -> dict:
        config = thread_config(user, conversation_id)
        try:
            result = self.agent.invoke(agent_input, config, context=user, version="v2")
        except ModelCallLimitExceededError:
            logger.warning(
                "thread=%s stopped at the model call limit", config["configurable"]["thread_id"]
            )
            return {
                "conversation_id": conversation_id,
                "reply": "That needed more steps than I can take in one message. "
                "Try asking for fewer things at once.",
                "pending_actions": [],
            }
        pending_actions = []
        for interrupt in result.interrupts:
            for action in interrupt.value["action_requests"]:
                pending_actions.append({"tool": action["name"], "summary": action["description"]})
        text = ""
        last_message = result.value["messages"][-1]
        if isinstance(last_message, AIMessage):
            text = last_message.text
        return {
            "conversation_id": conversation_id,
            "reply": text,
            "pending_actions": pending_actions,
        }


class MessageRateLimiter:
    def __init__(self, limit: int, window: timedelta, clock):
        self.limit = limit
        self.window = window
        self.clock = clock
        self.sent_at = defaultdict(deque)

    def allow(self, user_id: int) -> bool:
        now = self.clock()
        sent_at = self.sent_at[user_id]
        while sent_at and now - sent_at[0] >= self.window:
            sent_at.popleft()
        if len(sent_at) >= self.limit:
            return False
        sent_at.append(now)
        return True
