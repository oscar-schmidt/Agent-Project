import asyncio
import json
import re
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import os
import ollama
from Server.adaptor import Adaptor
from backend.model.states.graph_state.GraphState import GraphState
from backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from backend.tools.bind_tool.finalized_tool import finalized_tool
from backend.tools.command import command
from backend.tools.critique import critique
from backend.utils import get_user_input, log_decorator
from constants import SYSTEM_PROMPT_LIST
from backend.tools.get_tool_registry import get_tool_registry

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


async def tool_agent_async(state: GraphState, adaptor: Adaptor = None) -> GraphState:
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

    final_state = await invoke_tool(response.message, bind_tools, state, adaptor)
    return final_state


@log_decorator
async def tool_agent(state: GraphState, adaptor: Adaptor = None) -> GraphState:
    return await tool_agent_async(state, adaptor)


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


@log_decorator
async def invoke_tool(message, bind_tools, state: GraphState, adaptor: Adaptor = None) -> GraphState:
    tool_name_args = get_tool_name_list(message, state)
    user_input = get_user_input()
    new_state = state

    if not bind_tools or len(bind_tools) == 0 or not tool_name_args:
        if adaptor:
            new_state = await send_request_to_others(message, new_state, adaptor)
        else:
            new_state.messages.append(
                AIMessage(content="No tools available. Responding directly."))
        return new_state

    max_attempts = 2
    attempt = 0
    should_recall = True

    while attempt < max_attempts and should_recall:
        for tool_name, args in tool_name_args:
            new_state = await execute_tool(tool_name, args, bind_tools, new_state)

        finalized_result = await finalized_tool().ainvoke({"state": new_state})
        new_state = command("chat_tool", finalized_result)

        should_recall = critique(
            user_input, finalized_result.agent_response)

        if should_recall and adaptor:
            maybe_new_state = await send_request_to_others(message, new_state, adaptor)
            if isinstance(maybe_new_state, GraphState):
                new_state = maybe_new_state

        attempt += 1

    finalized_result = await finalized_tool().ainvoke({"state": new_state})
    final_state = command("chat_tool", finalized_result)
    return final_state


async def execute_tool(tool_name, args, bind_tools, state: GraphState) -> GraphState:
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

    return new_state


def get_tool_name_list(message, state: GraphState) -> list[tuple[str, dict]]:
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

            tools = content_json.get("tools", []) if content_json.get(
                "tools", []) else content_json.get("name", [])
            for t in tools:
                tool_name = t.get("tool_name") or t.get("tool")
                if not tool_name:
                    continue
                args = t.get("args", {})
                args["state"] = state
                tool_name_args.append((tool_name, args))
        except:
            state.messages.append(
                AIMessage(content="Can I Beg Your Pardon?"))

    return tool_name_args


async def send_request_to_others(message, state: GraphState, adaptor) -> GraphState:
    user_input = get_user_input()
    tool_registry = get_tool_registry()
    tool_names = list(tool_registry.keys())

    state.logs.append("[invoke_tool] No tools bound, querying directory agent")

    await adaptor.send_message({
        "message_type": "message",
        "recipient_id": "DirectoryAgent",
        "sender_id": adaptor.connection.agent_id,
        "message": user_input,
    }, timeout=15.0)

    # can we have another field to put real tool response?
    agent_response: ToolReturnClass = await adaptor.receive_message()

    # later i can filter the true agent response?
    if agent_response and agent_response.meta["tool_name"] in tool_names:
        response_content = agent_response.agent_response

        state.messages.append(AIMessage(content=response_content))
    else:
        state.messages.append(
            AIMessage(content="Unable to process request - no agents available")
        )

    return state
