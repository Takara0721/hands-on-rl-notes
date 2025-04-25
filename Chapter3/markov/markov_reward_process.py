from .markov import Markov
from Chapter3.state.base_states import BaseStates
from Chapter3.transition.state_transition import StateTransition
import numpy as np

class MarkovRewardProcess(Markov):
    def __init_transition_matrix(
            self,
            markov_chain: list[StateTransition]
    ) -> list[list[float]]:
        P = [[0.0 for _ in range(self.state_num)] for _ in range(self.state_num)]
        for state_transition in markov_chain:
            P[state_transition.start.index][state_transition.end.index] = state_transition.probability
        return P

    def __init__(
            self,
            states: type[BaseStates],
            markov_chain: list[StateTransition],
            gamma: float
    ) -> None:
        super().__init__(states, gamma)
        self.state_num = len(self.states)
        self.reward_list = [state.reward for state in self.states]
        self.transition_matrix = self.__init_transition_matrix(markov_chain)

    def compute_return(self, chain: list[BaseStates]) -> float:
        G = 0
        for state in reversed(chain):
            G = state.reward + G * self.gamma
        return G

    def compute_value_function(self) -> list[float]:
        rewards = np.array(self.reward_list).reshape((-1, 1))
        value_function = np.dot(
            np.linalg.inv(
                np.identity(self.state_num) - self.gamma * np.array(self.transition_matrix)
            ),
            rewards
        )
        return value_function