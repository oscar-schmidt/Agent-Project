import os
from dotenv import load_dotenv
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from ollama import chat
from langchain_core.messages import AIMessage

from agents.main_agent.backend.utils import log_decorator
from constants import SYSTEM_MESSAGE_LIST

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


def no_kb_agent(state: GraphState) -> GraphState:
    system_input = SYSTEM_MESSAGE_LIST.top_k_kb_not_found_prompt
    response = chat(
        OLLAMA_MODEL, [{"role": "system", "content": system_input}, {"role": "user", "content": "Please out put kb not found"}])
    state.messages.append(AIMessage(
        content=response.message.content))
    return state
