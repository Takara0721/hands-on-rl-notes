from Chapter3.state.base_states import BaseStates

class StateTransition:
    def __init__(self, start: BaseStates, end: BaseStates, probability: float) -> None:
        self.start = start
        self.end = end
        self.probability = probability
