import os
from dotenv import load_dotenv
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from ollama import chat
from langchain_core.messages import AIMessage
from agents.main_agent.backend.utils import get_user_input, log_decorator
from constants import SYSTEM_MESSAGE_LIST
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if LLM_PROVIDER == "openai":
    import openai
    openai.api_key = OPENAI_API_KEY
else:
    from ollama import chat


def rag_agent(state: GraphState) -> GraphState:
    user_input = get_user_input()

    prompt = SYSTEM_MESSAGE_LIST.top_k_kb_found_prompt.format(
        user_input=user_input,
        top_k_kb=state.qa_state.top_k_kb)

    if user_input:
        if LLM_PROVIDER == "openai":
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            content = response.choices[0].message.content
    else:
        response = chat(
            OLLAMA_MODEL,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]
        )
        content = response.message.content

    state.messages.append(
        AIMessage(content=content))
    state.logs.append(f"[rag_agent] prompt: {prompt}")
    return state
