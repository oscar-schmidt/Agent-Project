
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from agents.main_agent.backend.graph.summary_graph.build_summary_graph import build_summary_graph
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage


class summary_tool(BaseTool):
    """Return Final Summary"""
    name: str = "summary_tool"
    description: str = "a tool can summarize a knowledge base"
    def _run(self, arg: dict) -> None:
        pass

    async def _arun(self, arg: dict) -> str:
        state: GraphState = arg["state"]

        summary_state: GraphState = await self.subgraph.ainvoke(state)

        new_state = summary_state if isinstance(
            summary_state, GraphState) else GraphState(**summary_state)
        return ""
        return new_state["messages"][-1]
