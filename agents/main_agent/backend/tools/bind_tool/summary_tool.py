
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from agents.main_agent.backend.graph.summary_graph.build_summary_graph import build_summary_graph
from agents.main_agent.backend.tools.base_tool import BaseTool
from langchain_core.messages import HumanMessage, AIMessage


class summary_tool(BaseTool):
    """Return Final Summary"""

    def __init__(self) -> None:
        super().__init__()
        self.subgraph = build_summary_graph()

    async def ainvoke(self, arg: dict) -> ToolReturnClass:
        state: GraphState = arg["state"]

        summary_state: GraphState = await self.subgraph.ainvoke(state)

        new_state = summary_state if isinstance(
            summary_state, GraphState) else GraphState(**summary_state)

        return ToolReturnClass(
            state=new_state,
            agent_response=new_state.messages[-1].content if isinstance(
                new_state.messages[-1], AIMessage) else "No response",
            meta={"tool_name": "summary_tool"}
        )
