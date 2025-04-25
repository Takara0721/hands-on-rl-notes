import abc
from Chapter3.state.base_states import BaseStates

class Markov(abc.ABC):
    def __init__(self, states: type[BaseStates], gamma: float) -> None:
        self.states = states
        self.gamma = gamma

    @abc.abstractmethod
    def compute_return(self, chain: list[BaseStates]) -> float:
        pass