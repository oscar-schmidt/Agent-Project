import copy
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from agents.main_agent.backend.graph.qa_graph.build_qa_graph import build_qa_graph
from langchain_core.tools import BaseTool
from agents.main_agent.backend.utils import log_decorator
from langchain_core.messages import AIMessage, HumanMessage


@log_decorator
class qa_tool(BaseTool):
    name: str = "qa_tool"
    description: str = "a tool can search through a knowledge base if it exists"
    def _run(self, args: dict):
        pass
    async def _arun(self, args: dict) -> str:
        await build_qa_graph
        qa_state: GraphState = await self.subgraph.ainvoke(state)

        new_state = qa_state if isinstance(
            qa_state, GraphState) else GraphState(**qa_state)
        return ""
        return new_state["messages"][-1]
