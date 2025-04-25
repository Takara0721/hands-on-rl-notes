from Chapter3.action.base_actions import BaseActions
from Chapter3.state.base_states import BaseStates

class StateTransition:
    def __init__(self, start: BaseStates, end: BaseStates, probability: float, action: BaseActions = None) -> None:
        self.start = start
        self.end = end
        self.probability = probability
        self.action = action

    def __repr__(self) -> str:
        if self.action:
            return f"StateTransition(start={self.start.name}, end={self.end.name}, probability={self.probability})"
        else:
            return f"StateTransition(start={self.start.name}, action={self.action.name}, end={self.end.name}, probability={self.probability})"
