from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import os
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from agents.main_agent.backend.tools.base_tool import BaseTool
from agents.main_agent.backend.utils import get_user_input, log_decorator
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


class finalized_tool(BaseTool):

    async def ainvoke(self, args: dict) -> ToolReturnClass:
        return self.invoke(args)

    def invoke(self, args: dict) -> ToolReturnClass:
        state: GraphState = args["state"]
        user_input = get_user_input()

        recent_msgs = state.messages[-6:] if len(
            state.messages) > 3 else state.messages

        latest_tool_outputs = []
        if hasattr(state, "tool_outputs") and state.tool_outputs:
            latest_tool_outputs = [
                tool_response for tool_response in state.tool_outputs[-6:] if tool_response.get("agent_response")
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

        if LLM_PROVIDER == "openai":
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages
            )
            content = response.choices[0].message.content.strip()
        else:
            response = chat(OLLAMA_MODEL, messages)
            content = getattr(response.message, "content",
                              None) or "No response."

        state.messages.append(AIMessage(content=content))

        return ToolReturnClass(
            state=state,
            agent_response=content,
            meta={"tool_name": "finalized_tool"}
        )
