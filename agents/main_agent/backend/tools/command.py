from agents.main_agent.backend.model.states.StateManager import StateManager
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass
from agents.main_agent.backend.tools.get_tool_registry import get_tool_registry
from langchain_core.messages import AIMessage


def command(tool_name: str, result: ToolReturnClass) -> GraphState:
    tool_registry = get_tool_registry()
    state = StateManager.get_state()

    if not hasattr(state, "tool_outputs"):
        state.tool_outputs = []

    state.tool_outputs.append({
        "tool": tool_name,
        "agent_response": result.agent_response
    })

    state.logs.append(f"[command] Cached {tool_name} output")

    if tool_name == tool_registry["qa_tool"]["name"]:
        StateManager.update_substate("qa_state", result.state.qa_state)

        state.logs.append("[command] QA tool state updated")

    elif tool_name == tool_registry["summary_tool"]["name"]:
        StateManager.update_substate(
            "summary_state", result.state.summary_state)
        state.logs.append("[command] Summary tool state updated")

    elif tool_name == tool_registry["chat_tool"]["name"]:
        StateManager.update_substate(
            "messages", result.state.messages)
        state.logs.append("[command] Chat tool messages state updated")
        return state

    StateManager.update_substate("logs", result.state.logs)
    state.logs.append("[command] Log state updated")

    return state
