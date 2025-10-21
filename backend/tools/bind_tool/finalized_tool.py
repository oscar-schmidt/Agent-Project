from dotenv import load_dotenv
from ollama import chat
from langchain_core.messages import AIMessage, HumanMessage
import os
import streamlit
from backend.model.states.graph_state.GraphState import GraphState
from backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from backend.tools.base_tool import BaseTool
from backend.utils import get_user_input, log_decorator
from constants import SYSTEM_PROMPT_LIST

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


@log_decorator
class finalized_tool(BaseTool):
    """Respond to user queries"""

    def invoke(self, args: dict) -> ToolReturnClass:
        state: GraphState = args["state"]
        user_input = get_user_input()
        prompt = "You are an answer finalized agent. "
        final_tool_output = ""

        if hasattr(state, "tool_outputs"):
            for tool_output in state.tool_outputs:
                final_tool_output += f"\n[{tool_output['tool']}]: {tool_output['agent_response']}"

            prompt = SYSTEM_PROMPT_LIST.chat_tool_prompt.format(
                final_tool_output=final_tool_output)

        if not user_input and not final_tool_output:
            return ToolReturnClass(
                state=state,
                agent_response="No user input or tool output to process.",
                meta={"tool_name": "chat_tool"}
            )

        response = chat(
            OLLAMA_MODEL,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input or "Generate final answer based on tool outputs"}
            ]
        )

        state.messages.append(
            AIMessage(content=response.message.content))

        return ToolReturnClass(
            state=state,
            agent_response=state.messages.ai_response_list[-1].content if state.messages.ai_response_list else "No response",
            meta={"tool_name": "finalized_tool"}
        )
