from dotenv import load_dotenv
import os
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.utils import get_user_input, log_decorator
from constants import SYSTEM_PROMPT_LIST
from langchain_core.messages import AIMessage

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if LLM_PROVIDER == "openai":
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    from ollama import chat


def chat_agent(state: GraphState) -> GraphState:
    user_input = get_user_input()

    system_prompt = SYSTEM_PROMPT_LIST.default_prompt
    if getattr(state.summary_state, "final_summary", None):
        system_prompt = SYSTEM_PROMPT_LIST.final_summary_prompt.format(
            final_summary=state.summary_state.final_summary
        )

    if not user_input:
        return state

    if LLM_PROVIDER == "openai":
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        message_content = response.choices[0].message.content
    else:
        response = chat(
            OLLAMA_MODEL,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        message_content = response.message.content

    state.logs.append(system_prompt)
    state.messages.append(AIMessage(content=message_content))

    return state
