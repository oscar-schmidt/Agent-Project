import os
from dotenv import load_dotenv
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from ollama import chat
from langchain_core.messages import AIMessage
from agents.main_agent.backend.utils import log_decorator
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


async def no_kb_agent(state: GraphState) -> GraphState:
    system_input = SYSTEM_MESSAGE_LIST.top_k_kb_not_found_prompt
    user_input = "Please output kb not found"

    if LLM_PROVIDER == "openai":
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_input},
                {"role": "user", "content": user_input}
            ]
        )
        content = response.choices[0].message.content
    else:
        response = chat(
            OLLAMA_MODEL,
            [{"role": "system", "content": system_input},
             {"role": "user", "content": user_input}]
        )
        content = response.message.content

    state.tool_outputs.append({
        "tool": "no_kb_agent",
        "agent_response": content
    })
    return state
