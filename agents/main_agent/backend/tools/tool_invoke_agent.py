import asyncio
import json
import logging
import re
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import os
import ollama
from agents.main_agent.Server.adaptor import Adaptor
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from agents.main_agent.backend.tools.bind_tool.finalized_tool import finalized_tool
from agents.main_agent.backend.tools.command import command
from agents.main_agent.backend.tools.critique import critique
from agents.main_agent.backend.utils import get_user_input, log_decorator
from constants import SYSTEM_PROMPT_LIST
from agents.main_agent.backend.tools.get_tool_registry import get_tool_registry

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


async def tool_agent_async(state: GraphState) -> GraphState:
    user_input = get_user_input()
    bind_tools = get_bind_tools(state)

    system_prompt = SYSTEM_PROMPT_LIST.tool_router_prompt.format(
        user_input=user_input,
        tool_names={bind_tool["name"] for bind_tool in bind_tools} or "none"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    if LLM_PROVIDER == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": bind_tool["name"],
                    "description": bind_tool["description"],
                    "parameters": bind_tool["parameters"]
                }
            }
            for bind_tool in bind_tools
        ] if bind_tools else None

        completion_args = {
            "model": OPENAI_MODEL,
            "messages": messages,
        }

        if tools:
            completion_args["tools"] = tools
            completion_args["tool_choice"] = "auto"

        openai_response = await client.chat.completions.create(**completion_args)

        response_message = openai_response.choices[0].message

    else:
        client = ollama.AsyncClient()

        response = await client.chat(
            OLLAMA_MODEL,
            messages=messages,
            tools=[bind_tool["tool"] for bind_tool in bind_tools]
        )

        response_message = response.message

    final_state = await invoke_tool(response_message, bind_tools, state)
    return final_state


async def tool_agent(state: GraphState) -> GraphState:
    return await tool_agent_async(state)


def get_bind_tools(state: GraphState) -> list:
    tool_to_bind = []
    tool_registry = get_tool_registry()
    user_input = get_user_input()

    for tool_info in tool_registry.values():
        if LLM_PROVIDER == "ollama":
            if tool_info["condition"](state, user_input):
                tool_to_bind.append(tool_info)
        else:
            tool_to_bind.append(tool_info)

    tool_to_bind.sort(key=lambda t: t["priority"])

    state.logs.append(
        f"bind_tools: {[bind_tool['name'] for bind_tool in tool_to_bind]}")

    return tool_to_bind


async def invoke_tool(message, bind_tools, state: GraphState) -> GraphState:
    adaptor = None
    tool_name_args = get_tool_name_list(message, state)
    user_input = get_user_input()
    new_state = state
    arg_list = []

    # if not bind_tools or len(bind_tools) == 0 or not tool_name_args:
    #     if adaptor:
    #         new_state = await send_request_to_others(message, new_state)
    #     else:
    #         new_state.messages.append(
    #             AIMessage(content="No tools available. Responding directly."))
    #     return new_state

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

        if should_recall:
            maybe_new_state = await send_request_to_others(message, arg_list, new_state)
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
            state = await execute_tool(nested_name, nested_args, bind_tools, state)

    result = await tool_to_invoke["invoke"].ainvoke(args)
    new_state = command(tool_name, result)

    return new_state


def get_tool_name_list(message, state: GraphState) -> list[tuple[str, dict]]:
    tool_name_args = []
    if message.tool_calls:
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            raw_args = tool_call.function.arguments

            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"_raw": raw_args}

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


async def send_request_to_others(message, arg_list, state: GraphState) -> GraphState:
    tool_registry = get_tool_registry()
    tool_names = list(tool_registry.keys())
    prompt = prompt_generator(state)

    try:
        adaptor = await Adaptor.get_adaptor(
            agent_id="main_agent",
            description="Interactive client",
            capabilities=["qa", "summary"]
        )
    except Exception as e:
        logging.error(f"Adaptor init failed: {e}")

        if not getattr(adaptor, "_started", False):
            await adaptor.start()
            await asyncio.sleep(0.5)

        state.logs.append("[invoke_tool] Sending request to DirectoryAgent")

        if prompt["recipient_id"] and prompt["message"]:
            try:
                await adaptor.send_message({
                    "message_type": "message",
                    "recipient_id": prompt["recipient_id"],
                    "sender_id": "main_agent",
                    "message": prompt["message"],
                })
            except Exception as e:
                logging.error(f"Error sending message: {e}")

        try:
            agent_response: ToolReturnClass = await asyncio.wait_for(
                adaptor.receive_message(),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logging.error("Timeout waiting for response from DirectoryAgent")
        except Exception as e:
            logging.error(f"Receive failed: {e}")

            if agent_response and agent_response.meta.get("tool_name") in tool_names:
                response_content = agent_response.agent_response
            else:
                logging.error(f"{agent_response.agent_response}")

    return state
