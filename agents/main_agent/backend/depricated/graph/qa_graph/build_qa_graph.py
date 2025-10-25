from langgraph.graph import StateGraph, START, END
from agents.main_agent.backend.depricated.graph.get_excel_ready_pipeline import get_excel_ready_pipeline
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.nodes.utils.chat_agent import chat_agent
from agents.main_agent.backend.nodes.qa_node.no_kb_agent import no_kb_agent
from agents.main_agent.backend.nodes.qa_node.rag_agent import rag_agent
from agents.main_agent.backend.depricated.graph.get_pdf_ready_pipeline import get_pdf_ready_pipeline
from agents.main_agent.backend.depricated.graph.qa_graph.rag_router import rag_router
from agents.main_agent.backend.depricated.graph.qa_graph.upload_check_router import upload_check_router


def build_qa_graph(data_ready_node):

    graph = StateGraph(GraphState)

    graph.add_node("upload_check_router", lambda state: state)
    graph.add_node("get_data_ready_pipeline", data_ready_node)
    graph.add_node("router", lambda state: state)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("chat_agent", chat_agent)
    graph.add_node("no_kb_agent", no_kb_agent)

    graph.add_edge(START, "upload_check_router")
    graph.add_edge("get_data_ready_pipeline", "router")

    graph.add_conditional_edges(
        "router",
        rag_router,
        {
            "TRUE": "rag_agent",
            "FALSE": "chat_agent",
            "NO_KB": "no_kb_agent",
        },
    )
    graph.add_conditional_edges(
        "upload_check_router",
        upload_check_router,
        {
            "TRUE": "get_data_ready_pipeline",
            "FALSE": "chat_agent",
        },
    )

    graph.set_entry_point("upload_check_router")
    graph.set_finish_point("no_kb_agent")
    graph.set_finish_point("rag_agent")

    return graph.compile()
