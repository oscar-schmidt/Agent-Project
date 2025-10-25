from typing import Any, Optional, Union
from types import SimpleNamespace
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field


class MessageStore(BaseModel):
    messages: list[BaseMessage
                   ] = Field(default_factory=list)
    user_query_list: list[HumanMessage] = Field(default_factory=list)
    ai_response_list: list[AIMessage] = Field(default_factory=list)
    message_history: list[Union[HumanMessage, AIMessage]
                          ] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def append(self, message: Union[HumanMessage, AIMessage], save_to_history: bool = True):
        if message is None or not hasattr(message, "content"):
            return

        self.messages.append(message)

        if save_to_history:
            self.message_history.append(message)

        if isinstance(message, HumanMessage):
            self.user_query_list.append(message)
        elif isinstance(message, AIMessage):
            self.ai_response_list.append(message)

    def extend(self, messages: list[Any]):
        flat_msgs = MessageStore.flatten_messages(messages)
        for msg in flat_msgs:
            if not self._message_exists(msg):
                self.append(msg)

    @staticmethod
    def flatten_messages(msg: Any) -> list[Union[HumanMessage, AIMessage]]:
        flattened = []

        if msg is None:
            return flattened

        if isinstance(msg, (HumanMessage, AIMessage)):
            return [msg]

        if isinstance(msg, (list, tuple, set)):
            for item in msg:
                flattened.extend(MessageStore.flatten_messages(item))
            return flattened

        if isinstance(msg, SimpleNamespace) or isinstance(msg, dict):
            return flattened

        return flattened

    def _message_exists(self, message: Union[HumanMessage, AIMessage]) -> bool:
        if not hasattr(message, "content"):
            return False

        for m in self.messages:
            if getattr(m, "content", None) == message.content and type(m) == type(message):
                return True
        return False
