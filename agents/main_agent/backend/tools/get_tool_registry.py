from agents.main_agent.backend.tools.bind_tool.chat_tool import chat_tool
from agents.main_agent.backend.tools.bind_tool.qa_tool import qa_tool
from agents.main_agent.backend.tools.bind_tool.summary_tool import summary_tool


def get_tool_registry():
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
            "condition": lambda state, user_input: (
                any(kw in user_input.lower()
                    for kw in ["summary", "abstract", "overview"])
            ),
        },
        "chat_tool": {
            "tool": chat_tool,
            "invoke": chat_tool(),
            "name": "chat_tool",
            "priority": 3,
            "condition": lambda state, user_input: False
        }
    }
