from typing import Annotated
from typing_extensions import TypedDict, List, Any
from langgraph.graph.message import add_messages
from langgraph.channels import EphemeralValue


class State(TypedDict):
    """An agent state"""
    messages: Annotated[list, add_messages]
    plan: str
    critique: str


