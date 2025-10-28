from dotenv import load_dotenv
from ollama import chat
from langchain_core.messages import AIMessage, HumanMessage
import os
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from agents.main_agent.backend.tools.base_tool import BaseTool
from agents.main_agent.backend.utils import get_user_input, log_decorator
from constants import SYSTEM_PROMPT_LIST

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


@log_decorator
class finalized_tool(BaseTool):
    """Respond to user queries"""

    async def ainvoke(self, args: dict) -> ToolReturnClass:
        return self.invoke(args)

    def invoke(self, args: dict) -> ToolReturnClass:
        state: GraphState = args["state"]
        user_input = get_user_input()

        messages = []

        system_prompt = SYSTEM_PROMPT_LIST.finalized_tool_prompt

        messages.append({"role": "system", "content": system_prompt})

        for msg in state.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})

        if hasattr(state, "tool_outputs") and state.tool_outputs:
            for tool_output in state.tool_outputs:
                if tool_output.get("agent_response"):
                    messages.append({
                        "role": "user",
                        "content": f"[Tool Output: {tool_output['tool']}] {tool_output['agent_response']}"
                    })

        messages.append({
            "role": "user",
            "content": user_input or "Generate final answer based on context and tool outputs."
        })

        response = chat(OLLAMA_MODEL, messages)

        ai_message_content = response.message.content if response.message and response.message.content else "No response"
        state.messages.append(AIMessage(content=ai_message_content))

        return ToolReturnClass(
            state=state,
            agent_response=response.message.content if response.message.content else "No response",
            meta={"tool_name": "finalized_tool"}
        )
