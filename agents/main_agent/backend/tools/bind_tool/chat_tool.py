from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import os
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from agents.main_agent.backend.tools.base_tool import BaseTool
from agents.main_agent.backend.utils import get_user_input, log_decorator

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if LLM_PROVIDER == "openai":
    import openai
    openai.api_key = OPENAI_API_KEY
else:
    from ollama import chat


class chat_tool(BaseTool):
    """Respond to user queries"""

    async def ainvoke(self, args: dict) -> ToolReturnClass:
        state: GraphState = args["state"]
        user_input = get_user_input()

        if not user_input:
            return ToolReturnClass(
                state=state,
                agent_response="No user input",
                meta={"tool_name": "chat_tool"}
            )

        if LLM_PROVIDER == "openai":
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": user_input}]
            )
            answer = response.choices[0].message.content.strip()
        else:
            response = chat(
                OLLAMA_MODEL, [{"role": "user", "content": user_input}]
            )
            answer = response.message.content.strip()

        state.messages.append(AIMessage(content=answer))

        return ToolReturnClass(
            state=state,
            agent_response=answer if answer else "No response",
            meta={"tool_name": "chat_tool"}
        )
