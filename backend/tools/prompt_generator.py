import os
from dotenv import load_dotenv
from backend.model.states.graph_state.GraphState import GraphState
from constants import SYSTEM_PROMPT_LIST
from langchain_core.messages import HumanMessage, AIMessage
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


def prompt_generator(state: GraphState):
    system_prompt = SYSTEM_PROMPT_LIST.webSocket_prompt

    recent_msgs = state.messages[-6:] if len(
        state.messages) > 3 else state.messages

    messages = [
        {"role": "system", "content": system_prompt}]

    for msg in recent_msgs:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

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
