import os
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from backend.embedding.sql_setup import create_checkpoint_memory
from backend.model.states.graph_state.GraphState import GraphState
from backend.tools.tool_invoke_agent import tool_agent

load_dotenv()

SQL_PATH = os.getenv("SQL_PATH")

# Create a single memory instance to reuse across all graph invocations
_memory_instance = None
_compiled_graph = None


async def get_memory():
    """Initialize and return the memory checkpointer singleton"""
    global _memory_instance
    if _memory_instance is None:
        # Create a persistent connection
        conn = await aiosqlite.connect(SQL_PATH)
        _memory_instance = AsyncSqliteSaver(conn)
        await _memory_instance.setup()
    return _memory_instance


async def get_graph(state: GraphState = None):
    """Get or create the compiled graph with persistent memory"""
    global _compiled_graph

    if _compiled_graph is None:
        graph = StateGraph(GraphState)
        graph.add_node("tool_agent", tool_agent, is_async=True)

        graph.add_edge(START, "tool_agent")
        graph.add_edge("tool_agent", END)

        memory = await get_memory()
        _compiled_graph = graph.compile(checkpointer=memory)

    return _compiled_graph


async def cleanup():
    """Clean up resources when shutting down"""
    global _memory_instance
    if _memory_instance is not None and hasattr(_memory_instance, 'conn'):
        await _memory_instance.conn.close()
