import os
from dotenv import load_dotenv
from constants import SYSTEM_PROMPT_LIST
from ollama import chat

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


def critique(user_input: str, final_output: str) -> bool:
    system_prompt = SYSTEM_PROMPT_LIST.critique_prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_output},
        {"role": "user", "content": user_input}
    ]

    response = chat(OLLAMA_MODEL, messages)

    should_recall = str(response.message.content).strip().lower()
    if should_recall == 'true':
        return True
    else:
        return False
