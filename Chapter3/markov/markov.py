import abc
from Chapter3.state.base_state_enum import BaseStateEnum

class Markov(abc.ABC):
    def __init__(self, states: type[BaseStateEnum], gamma: float) -> None:
        self.states = states
        self.gamma = gamma

    @abc.abstractmethod
    def compute_return(self, chain: list[BaseStateEnum]) -> float:
        pass