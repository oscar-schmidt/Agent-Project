from abc import ABC, abstractmethod

from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass

"""
Not necessary to be used with Langchain Tools as  
"""
class BaseTool(ABC):
    @abstractmethod
    def ainvoke(self, args: dict) -> ToolReturnClass:
        pass
