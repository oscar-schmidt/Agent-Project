from typing import Annotated, Any, List
from pydantic import BaseModel, Field, root_validator
from agents.main_agent.backend.model.states.graph_state.ConfigState import ConfigState
from agents.main_agent.backend.model.states.graph_state.SummaryState import SummaryState
from agents.main_agent.backend.model.states.qa_state.DocState import DocState
from agents.main_agent.backend.model.stores.LogStore import LogStore
from agents.main_agent.backend.model.stores.MessageStore import MessageStore
from typing import Annotated
from langgraph.graph.message import add_messages



def merge_messages(current: MessageStore, new: MessageStore) -> MessageStore:
    if isinstance(current, dict):
        current = MessageStore(**current)

    if current is None:
        return new
    if new is None:
        return current

    all_formatted_msgs = []

    for msg in new.messages:
        msg_obj = msg[1] if isinstance(msg, tuple) else msg
        flattened = MessageStore.flatten_messages(msg_obj)
        all_formatted_msgs.extend(flattened)

    for m in all_formatted_msgs:
        if not current._message_exists(m):
            current.append(m)
    return current


def merge_logs(current: LogStore, new: LogStore) -> LogStore:
    if isinstance(current, dict):
        current = LogStore(**current)

    if current is None:
        return new
    if new is None:
        return current

    for log in new:
        if log not in current:
            current.append(log)
    return current


class GraphState(BaseModel):
    messages: Annotated[list, add_messages]
    logs: Annotated[LogStore, merge_logs] = Field(default_factory=LogStore)
    plan: str = ""


    model_config = {
        "arbitrary_types_allowed": True,
        "validate_default": False,
    }

    @root_validator(pre=True)
    def ensure_stores(cls, values):
        if values.get("messages") is None:
            values["messages"] = MessageStore()
        if values.get("logs") is None:
            values["logs"] = LogStore()
        return values
