import os
from dotenv import load_dotenv
from constants import SYSTEM_PROMPT_LIST

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


def critique(user_input: str, final_output: str) -> bool:
    system_prompt = SYSTEM_PROMPT_LIST.critique_prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_output},
        {"role": "user", "content": user_input}
    ]

    if LLM_PROVIDER == "openai":
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )
        message_content = response.choices[0].message.content
    else:
        response = chat(OLLAMA_MODEL, messages)
        message_content = response.message.content

    should_recall = str(message_content).strip().lower()
    if should_recall == 'true':
        return True
    else:
        return False
