from abc import ABC, abstractmethod

from backend.model.states.tool_state.ToolReturnClass import ToolReturnClass


class BaseTool(ABC):
    @abstractmethod
    def ainvoke(self, args: dict) -> ToolReturnClass:
        pass
