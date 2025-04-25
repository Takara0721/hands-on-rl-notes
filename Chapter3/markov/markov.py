import abc
from Chapter3.state.base_states import BaseStates


class Markov(abc.ABC):
    def __init__(self, states: BaseStates, gamma: float) -> None:
        self.states = states
        self.transition_matrix = None
        self.gamma = gamma

    @abc.abstractmethod
    def compute_return(self, chain: list[BaseStates]):
        pass

    @abc.abstractmethod
    def compute_value_function(self):
        pass