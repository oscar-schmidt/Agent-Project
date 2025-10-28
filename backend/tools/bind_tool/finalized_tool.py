from dotenv import load_dotenv
from ollama import chat
from langchain_core.messages import AIMessage, HumanMessage
import os
from backend.model.states.graph_state.GraphState import GraphState
from backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from backend.tools.base_tool import BaseTool
from backend.utils import get_user_input, log_decorator
from constants import SYSTEM_PROMPT_LIST

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


class finalized_tool(BaseTool):

    async def ainvoke(self, args: dict) -> ToolReturnClass:
        return self.invoke(args)

    def invoke(self, args: dict) -> ToolReturnClass:
        state: GraphState = args["state"]
        user_input = get_user_input()

        recent_msgs = state.messages[-3:] if len(
            state.messages) > 3 else state.messages

        latest_tool_outputs = []
        if hasattr(state, "tool_outputs") and state.tool_outputs:
            latest_tool_outputs = [
                tool_response for tool_response in state.tool_outputs[-2:] if tool_response.get("agent_response")
            ]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LIST.finalized_tool_prompt}]

        for msg in recent_msgs:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})

        for tool_output in latest_tool_outputs:
            tool_name = tool_output.get("tool", "unknown_tool")
            agent_response = tool_output.get("agent_response", "")
            messages.append({
                "role": "assistant",
                "content": f"[Tool Output - {tool_name}] {agent_response}"
            })

        messages.append({
            "role": "user",
            "content": user_input or "Please answer concisely based on the latest tool results."
        })

        response = chat(OLLAMA_MODEL, messages)
        content = getattr(response.message, "content", None) or "No response."

        state.messages.append(AIMessage(content=content))

        return ToolReturnClass(
            state=state,
            agent_response=content,
            meta={"tool_name": "finalized_tool"}
        )
