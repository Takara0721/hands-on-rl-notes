import numpy as np
from numpy import ndarray, dtype, float64

from .markov import Markov
from Chapter3.state.base_state_enum import BaseStateEnum
from Chapter3.action.base_action import BaseActionEnum, ActionDetail
from Chapter3.transition.state_transition import StateTransition
from Chapter3.base.state_action_pair import SAPair


class MarkovDecisionProcess(Markov):
    def __init_mrp_transition_matrix(self) -> list[list[float]]:
        mrp_transition_matrix = [[0.0 for _ in range(self.state_num)] for _ in range(self.state_num)]
        for start_state in self.states:
            for action in self.sa_pairs[start_state]:
                for end_state in self.sas_pairs[start_state][action]:
                    detail = self.state_action_details[SAPair(start_state, action)]
                    transition_probability = self.markov_chain[StateTransition(start_state, end_state, action)]
                    mrp_transition_matrix[start_state.index][end_state.index] += detail.probability * transition_probability

        return mrp_transition_matrix

    def __init_mrp_reward_list(self) -> list[float]:
        mrp_reward_list = [0.0 for _ in range(self.state_num)]
        for state in self.states:
            for action in self.sa_pairs[state]:
                detail = self.state_action_details[SAPair(state, action)]
                mrp_reward_list[state.index] += detail.reward * detail.probability

        return mrp_reward_list

    def __init_legal_sa_pairs(self) -> dict[BaseStateEnum, list[BaseActionEnum]]:
        sa_pairs = dict(zip(self.states, [[] for _ in range(self.state_num)]))
        for key in self.state_action_details.keys():
            sa_pairs[key.state].append(key.action)
        return sa_pairs

    def __init_legal_sas_pairs(self) -> dict[BaseStateEnum, dict[BaseActionEnum, list[BaseStateEnum]]]:
        sas_pairs = {
            state: {
                action:[]
                for action in self.sa_pairs[state]
            }
            for state in self.states
        }

        for key in self.markov_chain.keys():
            sas_pairs[key.start][key.action].append(key.end)
        return sas_pairs

    def __init__(
            self,
            states: type[BaseStateEnum],
            actions: type[BaseActionEnum],
            state_action_details: dict[SAPair, ActionDetail],
            gamma: float,
            markov_chain: dict[StateTransition, float]
    ) -> None:
        super().__init__(states, gamma)
        self.actions = actions
        self.state_action_details = state_action_details
        self.markov_chain = markov_chain
        self.state_num = len(self.states)
        self.action_num = len(self.actions)
        self.sa_pairs = self.__init_legal_sa_pairs()
        self.sas_pairs = self.__init_legal_sas_pairs()
        self.mrp_transition_matrix = None
        self.mrp_reward_list = None
        self.state_value_function_list = None

    def compute_return(self, chain: list[SAPair]) -> float:
        G = 0
        for sa_pair in reversed(chain):
            G = self.state_action_details[sa_pair].reward + G * self.gamma
        return G

    def compute_state_value_function_list(self) -> ndarray[tuple[int, int], dtype[float64]]:
        if self.mrp_reward_list is None and self.mrp_transition_matrix is None:
            self.mrp_transition_matrix = self.__init_mrp_transition_matrix()
            self.mrp_reward_list = self.__init_mrp_reward_list()
        rewards = np.array(self.mrp_reward_list).reshape((-1, 1))
        state_value_function_list = np.dot(
            np.linalg.inv(
                np.identity(self.state_num) - self.gamma * np.array(self.mrp_transition_matrix)
            ),
            rewards
        )
        self.state_value_function_list = state_value_function_list
        return state_value_function_list

    def compute_action_value_function(self, state: BaseStateEnum, action: BaseActionEnum) -> float:
        if self.state_value_function_list is None:
            self.compute_state_value_function_list()
        detail = self.state_action_details[SAPair(state, action)]
        action_value_function = detail.reward

        for next_state in self.sas_pairs[state][action]:
            action_value_function += (
                    self.gamma
                    * self.markov_chain[StateTransition(state, next_state, action)]
                    * float(self.state_value_function_list[next_state.index])
            )

        return action_value_function

    def sample(self, timestep_max: int) -> list[SAPair]:
        state_list = list(self.states)
        state = state_list[np.random.randint(self.state_num - 1)]
        end = state_list[-1]

        episode = []
        timestep = 0
        while state != end and timestep < timestep_max:
            action, rand, tmp = None, np.random.random(), 0
            for _action in self.sa_pairs[state]:
                tmp += self.state_action_details[SAPair(state, _action)].probability
                if rand < tmp:
                    action = _action
                    break

            episode.append(SAPair(state, action))
            rand, tmp = np.random.random(), 0
            for _next_state in self.sas_pairs[state][action]:
                tmp += self.markov_chain[StateTransition(state, _next_state, action)]
                if rand < tmp:
                    state = _next_state
                    break

            timestep += 1

        return episode

    def monte_carlo_policy_evaluation_v(
            self,
            timestep_max: int,
            sample_num: int
    ) -> ndarray[tuple[int, int], dtype[float64]]:
        state_value_function_list = np.zeros(self.state_num)
        count_list = np.zeros(self.state_num)

        for _ in range(sample_num):
            episode = self.sample(timestep_max)
            start = episode[0].state
            count_list[start.index] += 1
            state_value_function_list[start.index] += (
                    self.compute_return(episode)
                    - state_value_function_list[start.index]
            ) / count_list[start.index]

        return state_value_function_list.reshape(-1, 1)

    def occupancy_evaluation(
            self,
            state: BaseStateEnum,
            action: BaseActionEnum,
            timestep_max: int,
            sample_num: int
    ) -> float:
        pass
