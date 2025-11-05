import asyncio
import json
import logging
import re
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import os
import ollama
from agents.main_agent.Server.MainAgentAdaptor import MainAgentAdaptor
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from agents.main_agent.backend.tools.bind_tool.finalized_tool import finalized_tool
from agents.main_agent.backend.tools.command import command
from agents.main_agent.backend.tools.critique import critique
from agents.main_agent.backend.tools.prompt_generator import prompt_generator
from agents.main_agent.backend.utils import get_user_input
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
    tool_name_args = get_tool_name_list(message, state)
    user_input = get_user_input()
    new_state = state
    max_attempts = 2
    attempt = 0
    should_recall = True

    main_agent = await MainAgentAdaptor.create()
    while attempt < max_attempts and should_recall:
        for tool_name, args in tool_name_args:
            new_state = await execute_tool(tool_name, args, bind_tools, new_state)
        finalized_result = await finalized_tool().ainvoke({
            "state": new_state,
            "should_recall": should_recall
        })
        should_recall = critique(user_input, finalized_result.agent_response)
        # should_recall = False
        if should_recall:
            format_data = prompt_generator(finalized_result.agent_response)
            directory_message = format_data.get("message")
            await main_agent.send_message(recipient_id="DirectoryAgent", message=directory_message)
            # await main_agent.send_message(recipient_id="WebAgent", message=directory_message)
            directory_response = await main_agent.receive_message()
            if not directory_response:
                logging.info("No response from DirectoryAgent.")
                break
            logging.info("Response from DirectoryAgent: {directory_response}")
            try:
                dir_data = json.loads(directory_response)
                format_data = prompt_generator(dir_data)
                recipient_id = format_data.get("recipient_id")
                sender_id = format_data.get("sender_id")
                next_message = format_data.get("message")
            except json.JSONDecodeError:
                logging.info("DirectoryAgent returned invalid JSON.")
                break

            await main_agent.send_message(recipient_id=recipient_id, message=next_message)
            # await main_agent.send_message(recipient_id="WebAgent", message=next_message)
            agent_response = await main_agent.receive_message()

            finalized_result.agent_response = agent_response
            agent_obj = json.loads(agent_response)
            logging.info(
                f"Response from {agent_obj['sender_id']}: {agent_response}")
            new_state.tool_outputs.append({
                "tool": agent_obj["sender_id"],
                "agent_response": agent_obj["message"]})

        attempt += 1

    new_state.tool_outputs = [
        tool for tool in new_state.tool_outputs if tool.get("tool") != "finalized_tool"
    ]

    finalized_result = await finalized_tool().ainvoke({"state": new_state, "should_recall": False})
    return finalized_result.state


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
