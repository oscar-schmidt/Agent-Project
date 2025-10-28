from attr import dataclass
from backend.model.states.graph_state.GraphState import GraphState
from backend.model.states.tool_state.ToolMetaDict import ToolMetaDict


@dataclass
class ToolReturnClass:
    state: GraphState
    agent_response: str = ""
    meta: ToolMetaDict = ToolMetaDict()
