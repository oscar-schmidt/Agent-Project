import os
import sqlite3
import json
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.utils import log_decorator

load_dotenv()

SQL_PATH = os.getenv("SQL_PATH")
read_config = {"configurable": {"thread_id": "123"}}


def create_checkpoint_memory():
    conn = sqlite3.connect(SQL_PATH, check_same_thread=False)
    return SqliteSaver(conn)
