import numpy as np

from .markov import Markov
from Chapter3.state.base_states import BaseStates
from Chapter3.action.base_actions import BaseActions
from Chapter3.reward.reward import Reward
from Chapter3.transition.state_transition import StateTransition

class MarkovDecisionProcess(Markov):
    def __get_reward(self, state: BaseStates, action: BaseActions) -> float:
        for reward in self.rewards:
            if reward.state == state and reward.action == action:
                return reward.value
        raise Exception("非合法数据！")

    def __init_mrp_transition_matrix(
            self,
            policy: list[tuple[BaseStates, BaseActions, float]],
            markov_chain: list[StateTransition]
    ) -> list[list[float]]:
        pass

    def __init_mrp_reward_list(
            self,
            policy: list[tuple[BaseStates, BaseActions, float]],
            rewards: list[Reward]
    ) -> list[float]:
        pass

    def __init__(
            self,
            states: type[BaseStates],
            actions: type[BaseActions],
            policy: list[tuple[BaseStates, BaseActions, float]],
            rewards: list[Reward],
            gamma: float,
            markov_chain: list[StateTransition]
    ) -> None:
        super().__init__(states, gamma)
        self.actions = actions
        self.policy = policy
        self.rewards = rewards
        self.markov_chain = markov_chain
        self.state_num = len(self.states)
        self.action_num = len(self.actions)
        self.mrp_transition_matrix = None
        self.mrp_reward_list = None

    def compute_return(self, chain: list[tuple[BaseStates, BaseActions]]) -> float:
        G = 0
        for state, action in reversed(chain):
            G = self.__get_reward(state, action) + G * self.gamma
        return G

    def compute_state_value_function(self) -> list[float]:
        if not self.mrp_reward_list and not self.mrp_transition_matrix:
            self.mrp_transition_matrix = self.__init_mrp_transition_matrix()
            self.mrp_reward_list = self.__init_mrp_reward_list(self.policy, self.rewards)
        rewards = np.array(self.mrp_reward_list).reshape((-1, 1))
        state_value_function = np.dot(
            np.linalg.inv(
                np.identity(self.state_num) - self.gamma * np.array(self.mrp_transition_matrix)
            ),
            rewards
        )
        return state_value_function

    def compute_action_value_function(self) -> list[float]:
        pass