import asyncio
import json
import re
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import os
import ollama
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.tools.bind_tool.finalized_tool import finalized_tool
#from agents.main_agent.backend.tools.command import command
from agents.main_agent.backend.utils import get_user_input, log_decorator
from constants import SYSTEM_PROMPT_LIST
#from agents.main_agent.backend.tools.get_tool_registry import get_tool_registry
from langchain_core.messages import ToolMessage
load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

"""No longer necssary the main agent should just be able to invoke tools based on user input and tool availability"""
async def tool_agent_async(state: GraphState) -> GraphState:
    client = ollama.AsyncClient()
    user_input = get_user_input()

    bind_tools = get_bind_tools(state)

    system_prompt = SYSTEM_PROMPT_LIST.tool_router_prompt.format(
        user_input=user_input, tool_names={bind_tool["name"] for bind_tool in bind_tools} or "none")

    response = await client.chat(
        OLLAMA_MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_input}],
        tools=[bind_tool["tool"] for bind_tool in bind_tools],
    )

    state.logs.append(f"{response}")

    final_state = await invoke_tool(response.message, bind_tools, state)
    return final_state


@log_decorator
async def tool_agent(state: GraphState) -> GraphState:
    return await tool_agent_async(state)

"""No longer need to bind tools becuase we want to allow the llm to decide which tools to use based on the user input"""
@log_decorator
def get_bind_tools(state: GraphState) -> list:
    tool_to_bind = []
    tool_registry = get_tool_registry()
    user_input = get_user_input()

    for tool_info in tool_registry.values():
        if tool_info["condition"](state, user_input):
            tool_to_bind.append(tool_info)

    tool_to_bind.sort(key=lambda t: t["priority"])

    state.logs.append(
        f"bind_tools: {[bind_tool['name'] for bind_tool in tool_to_bind]}")

    return tool_to_bind





class ToolNode:
    """A node that runs the tools requested however not in uses becuase using the provided tool node from langgraph"""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: GraphState):
        import datetime
        date = datetime.datetime.now()
        formatted_date = date.strftime("%m/%d/%Y %H:%M:%S")
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_name,
                tool_call_id=tool_call["id"],
            )
        )
        return str(formatted_date)

    async def invoke_tool(self, message, bind_tools, state: GraphState) -> GraphState:
        tool_name_args = self.get_tool_name_list(message, state)
        new_state = state

        for tool_name, args in tool_name_args:
            new_state = await self.execute_tool(tool_name, args, bind_tools, new_state)

        finalized_result = await finalized_tool().ainvoke({"state": new_state})

        final_state = command("chat_tool", finalized_result)

        return final_state

    async def execute_tool(self, tool_name, args, bind_tools, state: GraphState) -> GraphState:
        tool_to_invoke = next(
            (t for t in bind_tools if t["name"] == tool_name), None)

        if tool_to_invoke is None:
            state.logs.append(f"[execute_tool] Tool {tool_name} not found")
            return state

        if "tools" in args and isinstance(args["tools"], list):
            nested_tools = args["tools"]
            for nested in nested_tools:
                nested_name = nested.get("tool_name")
                if nested_name == tool_name:
                    state.logs.append(
                        f"[execute_tool] Skipped self-nesting: {tool_name}")
                    continue
                nested_args = nested.get("args", {})
                nested_args["state"] = state
                state = execute_tool(nested_name, nested_args, bind_tools, state)

        result = await tool_to_invoke["invoke"].ainvoke(args)
        new_state = command(tool_name, result)
        state.logs.append(f"[execute_tool] Executing {tool_name}")

        return new_state

    def get_tool_name_list(self, message, state: GraphState) -> list[tuple[str, dict]]:
        tool_name_args = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                args = tool_call.function.arguments or {}
                args["state"] = state
                tool_name_args.append((tool_name, args))
        else:
            try:
                try:
                    content_json = json.loads(message.content)
                except json.JSONDecodeError:
                    fixed_content = message.content.replace("'", '"')
                    content_json = json.loads(fixed_content)

                tools = content_json.get("tools", [])
                for t in tools:
                    tool_name = t.get("tool_name") or t.get("tool")
                    if not tool_name:
                        continue
                    args = t.get("args", {})
                    args["state"] = state
                    tool_name_args.append((tool_name, args))
            except:
                state.messages.ai_response_list.append(
                    AIMessage(content="Can I Beg Your Pardon?"))

        return tool_name_args

