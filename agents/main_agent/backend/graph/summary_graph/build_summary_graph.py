import os
import sqlite3
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.nodes.summary_node.get_summary_node import get_summary_node
from agents.main_agent.backend.nodes.utils.chat_agent import chat_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

SQL_PATH = os.getenv("SQL_PATH")


def build_summary_graph():

    graph = StateGraph(GraphState)

    graph.add_node("get_summary_node", get_summary_node)
    graph.add_node("summary_chat_agent", chat_agent)

    graph.add_edge(START, "get_summary_node")
    graph.add_edge("get_summary_node", "summary_chat_agent")
    graph.add_edge("get_summary_node", END)

    graph.set_entry_point("get_summary_node")
    graph.set_finish_point("summary_chat_agent")

    return graph.compile()
