from dotenv import load_dotenv
from ollama import chat
from langchain_core.messages import AIMessage, HumanMessage
import os
from backend.model.states.graph_state.GraphState import GraphState
from backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from backend.tools.base_tool import BaseTool
from backend.utils import get_user_input, log_decorator

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


@log_decorator
class chat_tool(BaseTool):
    """Respond to user queries"""

    async def ainvoke(self, args: dict) -> ToolReturnClass:
        state: GraphState = args["state"]
        user_input = get_user_input()

        if user_input:
            response = chat(
                OLLAMA_MODEL, [{"role": "user", "content": user_input}])

        state.messages.append(
            AIMessage(content=response.message.content))

        return ToolReturnClass(
            state=state,
            agent_response=response.message.content if response.message.content else "No response",
            meta={"tool_name": "chat_tool"}
        )
