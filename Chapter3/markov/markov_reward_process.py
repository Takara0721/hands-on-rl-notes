from .markov import Markov
from Chapter3.state.base_states import BaseStates


class MarkovRewardProcess(Markov):
    def __init__(self, states: BaseStates, gamma: float):
        super().__init__(states, gamma)

    def compute_return(self, chain: list[BaseStates]):
        pass

    def compute_value_function(self):
        pass