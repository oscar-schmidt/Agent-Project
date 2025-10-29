from backend.tools.bind_tool.chat_tool import chat_tool
from backend.tools.bind_tool.qa_tool import qa_tool
from backend.tools.bind_tool.summary_tool import summary_tool


def get_tool_registry():
    return {
        "qa_tool": {
            "tool": qa_tool,
            "invoke": qa_tool(),
            "name": "qa_tool",
            "description": "Perform RAG-based question answering",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User question"},
                    "collection_name": {
                        "type": "string",
                        "description": "Optional: specific Chroma collection to query"
                    }
                },
                "required": ["query"]
            },
            "priority": 1,
            # "condition": lambda state, user_input: True
        },
        "summary_tool": {
            "tool": summary_tool,
            "invoke": summary_tool(),
            "name": "summary_tool",
            "description": "Generate document summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "document": {
                        "type": "string",
                        "description": "Document text to summarize (optional if using collection)"
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Chroma collection name (optional if document provided)"
                    },
                    "summary_type": {
                        "type": "string",
                        "enum": ["brief", "detailed", "bullet_points"],
                        "description": "Type of summary to generate"
                    }
                },
                "required": []
            },
            "priority": 2,
            # "condition": lambda state, user_input: (
            #     any(kw in user_input.lower()
            #         for kw in ["summary", "abstract", "overview"])
            # ),
        },
        "chat_tool": {
            "tool": chat_tool,
            "invoke": chat_tool(),
            "description": "General conversation when no specific tool is needed",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "User's message"
                    }
                },
                "required": ["message"]
            },
            "name": "chat_tool",
            "priority": 3,
            # "condition": lambda state, user_input: False
        }
    }
