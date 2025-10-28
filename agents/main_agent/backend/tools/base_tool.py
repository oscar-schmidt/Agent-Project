from abc import ABC, abstractmethod

from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass


class BaseTool(ABC):
    @abstractmethod
    def ainvoke(self, args: dict) -> ToolReturnClass:
        pass
