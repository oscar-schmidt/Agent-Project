from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.nodes.pdf_node.chunk_pdf_node import chunk_pdf_node
from agents.main_agent.backend.utils import log_decorator


@log_decorator
def get_pdf_ready_pipeline(state: GraphState) -> GraphState:
    if getattr(state.qa_state, "is_upload") and not getattr(state.qa_state, "is_processed"):
        chunk_pdf_node(state)
        state.qa_state.is_processed = True
    return state
