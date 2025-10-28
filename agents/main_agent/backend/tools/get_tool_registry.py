from agents.main_agent.backend.tools.bind_tool.communicate_tool import communicate_tool
from agents.main_agent.backend.tools.bind_tool.qa_tool import qa_tool
from agents.main_agent.backend.tools.bind_tool.summary_tool import summary_tool


def get_tool_registry(websocket):
    return {
        "qa_tool": {
            "tool": qa_tool,
            "invoke": qa_tool(),
            "name": "qa_tool",
            "priority": 1,
            "condition": lambda state, user_input: True
        },
        "summary_tool": {
            "tool": summary_tool,
            "invoke": summary_tool(),
            "name": "summary_tool",
            "priority": 2,
            "condition": lambda state, user_input: True
        },
        "communicate_tool": {
            "tool": communicate_tool,
            "invoke": communicate_tool(websocket=websocket),
            "name": "communicate_tool",
            "priority": 3,
            "condition": lambda state, user_input: True
        }
    }
