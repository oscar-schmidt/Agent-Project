from typing import Annotated, Any
from pydantic import BaseModel, Field, root_validator
from backend.model.states.graph_state.SummaryState import SummaryState
from backend.model.states.qa_state.DocState import DocState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages


class GraphState(BaseModel):
    qa_state: DocState = Field(
        default_factory=DocState)
    summary_state: SummaryState = Field(default_factory=SummaryState)
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = []

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_default": False,
    }
