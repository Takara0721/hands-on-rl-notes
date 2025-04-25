from .markov import Markov
from Chapter3.state.base_states import BaseStates
from Chapter3.transition.state_transition import StateTransition

class MarkovRewardProcess(Markov):
    def __init_transition_matrix(
            self,
            states: type[BaseStates],
            markov_chain: list[StateTransition]
    ) -> list[list[float]]:
        n = len(states)
        P = [[0.0 for _ in range(n)] for _ in range(n)]
        for state_transition in markov_chain:
            P[state_transition.start.index][state_transition.end.index] = state_transition.probability
        return P

    def __init_reward_list(self, states: type[BaseStates]) -> list[float]:
        reward_list = []
        for state in states:
            reward_list.append(state.reward)
        return reward_list

    def __init__(
            self,
            states: type[BaseStates],
            markov_chain: list[StateTransition],
            gamma: float
    ) -> None:
        super().__init__(states, gamma)
        self.transition_matrix = self.__init_transition_matrix(states, markov_chain)
        self.reward_list = self.__init_reward_list(states)

    def compute_return(self, chain: list[BaseStates]) -> float:
        G = 0
        for state in reversed(chain):
            G = state.reward + G * self.gamma
        return G

    def compute_value_function(self) -> list[float]:
        pass