from .markov import Markov
from Chapter3.state.base_states import BaseStates
import enum

class MarkovDecisionProcess(Markov):
    def __init__(self, states: type[BaseStates], gamma: float) -> None:
        super().__init__(states, gamma)

    def compute_return(self, chain: list[BaseStates]) -> float:
        pass

