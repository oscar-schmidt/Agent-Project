from dotenv import load_dotenv
from ollama import chat
from langchain_core.messages import AIMessage, HumanMessage
import os
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.utils import get_user_input, log_decorator
from constants import SYSTEM_PROMPT_LIST

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


@log_decorator
def chat_agent(state: GraphState) -> GraphState:
    user_input = get_user_input()

    system_prompt = SYSTEM_PROMPT_LIST.default_prompt

    if getattr(state.summary_state, "final_summary", None):
        system_prompt = SYSTEM_PROMPT_LIST.final_summary_prompt.format(
            final_summary=state.summary_state.final_summary)

    if user_input:
        response = chat(
            OLLAMA_MODEL, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}])

        state.logs.append(system_prompt)
        state.logs.append(response)
        state.messages.append(
            AIMessage(content=response.message.content))

    return state
