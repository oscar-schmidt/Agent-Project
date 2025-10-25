import copy
from agents.main_agent.backend.depricated.graph.get_excel_ready_pipeline import get_excel_ready_pipeline
from agents.main_agent.backend.depricated.graph.get_pdf_ready_pipeline import get_pdf_ready_pipeline
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from agents.main_agent.backend.depricated.graph.qa_graph.build_qa_graph import build_qa_graph
from agents.main_agent.backend.tools.base_tool import BaseTool
from agents.main_agent.backend.utils import log_decorator


@log_decorator
class qa_tool(BaseTool):

    def __init__(self):
        super().__init__()

    async def ainvoke(self, args: dict) -> ToolReturnClass:
        state: GraphState = args["state"]
        doc_name = state.qa_state.doc_name.lower()

        if doc_name.endswith(".pdf"):
            subgraph = build_qa_graph(get_pdf_ready_pipeline)
        elif doc_name.endswith(".xlsx"):
            subgraph = build_qa_graph(get_excel_ready_pipeline)
        else:
            raise ValueError(f"Unsupported file type: {doc_name}")

        state_for_invoke = copy.deepcopy(state)
        qa_state: GraphState = await subgraph.ainvoke(state_for_invoke)

        new_state = qa_state if isinstance(
            qa_state, GraphState) else GraphState(**qa_state)

        return ToolReturnClass(
            state=new_state,
            agent_response=(
                new_state.messages.ai_response_list[-1].content
                if new_state.messages.ai_response_list
                else "No response"
            ),
            meta={"tool_name": "qa_tool"},
        )
