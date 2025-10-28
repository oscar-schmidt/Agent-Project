import copy
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from agents.main_agent.backend.graph.qa_graph.build_qa_graph import build_qa_graph
from agents.main_agent.backend.tools.base_tool import BaseTool
from agents.main_agent.backend.utils import log_decorator
from langchain_core.messages import AIMessage, HumanMessage


class qa_tool(BaseTool):

    def __init__(self):
        super().__init__()
        self.subgraph = build_qa_graph()

    async def ainvoke(self, args: dict) -> ToolReturnClass:
        state: GraphState = args["state"]

        qa_state: GraphState = await self.subgraph.ainvoke(state)

        new_state = qa_state if isinstance(
            qa_state, GraphState) else GraphState(**qa_state)

        return ToolReturnClass(
            state=new_state,
            agent_response=(
                new_state.messages[-1].content
                if isinstance(new_state.messages[-1], AIMessage)
                else "No response"
            ),
            meta={"tool_name": "qa_tool"},
        )
