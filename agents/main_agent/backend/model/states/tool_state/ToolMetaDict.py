from typing import TypedDict


class ToolMetaDict(TypedDict):
    tool_call_id: str
    tool_name: str