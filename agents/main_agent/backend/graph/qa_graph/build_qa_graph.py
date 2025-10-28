from langgraph.graph import StateGraph, START, END
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.nodes.utils.chat_agent import chat_agent
from agents.main_agent.backend.nodes.qa_node.no_kb_agent import no_kb_agent
from agents.main_agent.backend.nodes.qa_node.rag_agent import rag_agent
from agents.main_agent.backend.nodes.qa_node.rag_router import rag_router


def build_qa_graph():

    graph = StateGraph(GraphState)

    graph.add_node("router", lambda state: state)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("chat_agent", chat_agent)
    graph.add_node("no_kb_agent", no_kb_agent)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        rag_router,
        {
            "TRUE": "rag_agent",
            "FALSE": "chat_agent",
            "NO_KB": "no_kb_agent",
        },
    )

    graph.set_entry_point("router")
    graph.set_finish_point("no_kb_agent")
    graph.set_finish_point("rag_agent")

    return graph.compile()
