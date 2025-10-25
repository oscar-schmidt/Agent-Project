from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.stores.LogStore import LogStore
from agents.main_agent.backend.model.stores.MessageStore import MessageStore


class StateManager():
    state: GraphState = None

    @classmethod
    def get_state(cls) -> GraphState:
        if cls.state is None:
            cls.state = GraphState()
        return cls.state

    @classmethod
    def update_state(cls, new_state) -> GraphState:
        if isinstance(new_state, dict):
            cls._state = GraphState(**new_state)
        else:
            cls._state = new_state

    @classmethod
    def update_substate(cls, state_name: str, new_value):
        state = cls.get_state()

        if state_name == "messages":
            if isinstance(new_value, MessageStore):
                state.messages.extend(new_value)
            elif isinstance(new_value, list):
                state.messages.extend(new_value)
            else:
                raise TypeError(
                    f"Invalid type for messages: {type(new_value)}")

        elif state_name == "logs":
            if isinstance(new_value, list):
                state.logs.extend(new_value)
            elif isinstance(new_value, str):
                state.logs.append(new_value)
            elif isinstance(new_value, LogStore):
                state.logs.extend(new_value)
            else:
                raise TypeError(f"Invalid type for logs: {type(new_value)}")

        else:
            setattr(state, state_name, new_value)

    @staticmethod
    def _message_exists(messages, msg: MessageStore) -> bool:
        return any(message.content == msg.content and type(message) == type(msg) for message in messages)
