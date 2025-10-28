from attr import dataclass
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolMetaDict import ToolMetaDict


@dataclass
class ToolReturnClass:
    state: GraphState
    agent_response: str = ""
    meta: ToolMetaDict = ToolMetaDict()
