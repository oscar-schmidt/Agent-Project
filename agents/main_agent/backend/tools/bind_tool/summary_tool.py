
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from agents.main_agent.backend.depricated.graph.qa_graph.build_qa_graph import build_qa_graph
from agents.main_agent.backend.depricated.graph.get_pdf_ready_pipeline import get_pdf_ready_pipeline
from agents.main_agent.backend.depricated.graph.summary_graph.build_summary_graph import build_summary_graph
from agents.main_agent.backend.tools.base_tool import BaseTool



class summary_tool(BaseTool):
    """Return Final Summary"""

    def __init__(self) -> None:
        super().__init__()
        self.subgraph = build_summary_graph()

    async def ainvoke(self, arg: dict) -> ToolReturnClass:
        state: GraphState = arg["state"]

        if not state.qa_state.is_processed:
            get_pdf_ready_pipeline(state)

        summary_state: GraphState = self.subgraph.ainvoke(state)
        summary_state: GraphState = await self.subgraph.ainvoke(state)

        new_state = summary_state if isinstance(
            summary_state, GraphState) else GraphState(**summary_state)

        return ToolReturnClass(
            state=new_state,
            agent_response=new_state.messages.ai_response_list[-1].content if new_state.messages.ai_response_list else "No response",
            meta={"tool_name": "summary_tool"}
        )
