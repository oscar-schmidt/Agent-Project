import json
from langchain_core.messages import ToolMessage
from common.agent import AgentState as State

class ToolNode:
    """A node that runs the tools requested however not in uses becuase using the provided tool node from langgraph"""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State):
        if not (messages := state.get("messages", [])):
            raise ValueError("No message found in state")

        last_message = messages[-1]
        if not hasattr(last_message, 'tool_calls'):
            return

        outputs = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_to_call = self.tools_by_name.get(tool_name)
            tool_result = None

            if not tool_to_call:
                tool_result = f"Error: Tool '{tool_name}' not found."
            else:
                try:
                    if tool_name == "ContactOtherAgents":
                        connection = state.get("_websocket_connection")
                        print(type(connection))
                        if not connection:
                            tool_result = "Error: WebSocket connection not found in state."
                        else:
                            tool_args["_websocket_connection"] = connection
                            tool_result = tool_to_call.invoke(tool_args)
                    else:
                        tool_result = tool_to_call.invoke(tool_args)
                except Exception as e:
                    tool_result = f"Error running tool '{tool_name}': {e}"

            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        state["_websocket_connection"] = ""
        print(outputs)
        return {"messages": outputs}

