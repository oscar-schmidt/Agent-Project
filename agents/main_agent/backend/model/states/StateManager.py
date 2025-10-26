from backend.model.states.graph_state.GraphState import GraphState


class StateManager:
    _state: GraphState = None

    @classmethod
    def get_state(cls) -> GraphState:
        if cls._state is None:
            cls._state = GraphState()
        return cls._state

    @classmethod
    def update_state(cls, new_state):
        if hasattr(cls, "_state") and cls._state:
            new_state.messages = cls._state.messages + [
                m for m in new_state.messages if m not in cls._state.messages
            ]
        cls._state = new_state

    @classmethod
    def update_substate(cls, state_name: str, new_value):
        state = cls.get_state()
        setattr(state, state_name, new_value)
