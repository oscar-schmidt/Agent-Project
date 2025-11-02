import os
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.tools.tool_invoke_agent import tool_agent

load_dotenv()

SQL_PATH = os.getenv("SQL_PATH")

_memory_instance = None


async def get_memory():
    global _memory_instance
    if _memory_instance is None:
        conn = await aiosqlite.connect("db/main_agent.db")
        _memory_instance = AsyncSqliteSaver(conn)
        await _memory_instance.setup()
    return _memory_instance


async def get_graph():

    graph = StateGraph(GraphState)

    graph.add_node("tool_agent", tool_agent, is_async=True)

    graph.add_edge(START, "tool_agent")
    graph.add_edge("tool_agent", END)

    memory = await get_memory()
    _compiled_graph = graph.compile(checkpointer=memory)

    return _compiled_graph
