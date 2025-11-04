import os
from dotenv import load_dotenv
from agents.main_agent.backend.model.states.StateManager import StateManager
from constants import SYSTEM_PROMPT_LIST
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
import json

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


def prompt_generator(agent_response: str | dict = None):
    state = StateManager.get_state()
    system_prompt = SYSTEM_PROMPT_LIST.webSocket_prompt

    recent_msgs = state.messages[-6:] if len(
        state.messages) > 6 else state.messages

    messages = [{"role": "system", "content": str(system_prompt)}]

    for msg in recent_msgs:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": str(msg.content)})

    if agent_response is not None:
        if isinstance(agent_response, dict):
            content_str = json.dumps(agent_response)
        else:
            content_str = str(agent_response)
        messages.append({"role": "assistant", "content": content_str})

    try:
        if LLM_PROVIDER == "openai":
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages
            )
            message_content = response.choices[0].message.content
        else:
            response = chat(OLLAMA_MODEL, messages)
            message_content = response.message.content

        try:
            return json.loads(message_content)
        except json.JSONDecodeError:
            return {
                "recipient_id": "DirectoryAgent",
                "message": f"Invalid JSON from LLM: {message_content}"
            }

    except Exception as e:
        return {
            "recipient_id": "DirectoryAgent",
            "message": f"LLM call failed: {e}"
        }
