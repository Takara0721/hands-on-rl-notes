import copy
import numpy as np
from .base_iteration import BaseIteration
from gymnasium.core import Env
from Chapter4.env.my_cliff_walking_env import MyCliffWalkingEnv

class PolicyIteration(BaseIteration):
    def __init_end_states(self) -> set[int]:
        end_states = set()
        for state in self.env.P:
            for action in self.env.P[state]:
                for transition_prob in self.env.P[state][action]:
                    if transition_prob[3]:
                        end_states.add(transition_prob[1])

        return end_states

    def __get_state_policy(
            self,
            np_list: np.ndarray
    ) -> list[float]:
        max_condition = (np_list == np.max(np_list))
        max_num = np.sum(max_condition)
        state_policy = max_condition.astype(float)
        state_policy /= max_num
        return state_policy.tolist()

    def __init__(self, env: Env | MyCliffWalkingEnv, theta: float, gamma: float) -> None:
        super().__init__(env, theta, gamma)
        self.value_function_list = np.zeros(len(env.P))
        self.policy = [[1 / len(env.P[i]) for _ in range(len(env.P[i]))] for i in range(len(env.P))]
        self.end_states = self.__init_end_states()

    def policy_evaluation(self) -> None:
        cnt = 1
        while True:
            max_diff = 0
            next_value_function_list = np.zeros(len(self.env.P))
            for state in self.env.P:
                if state not in self.end_states:
                    for action in self.env.P[state]:
                        for p, next_state, reward, terminated in self.env.P[state][action]:
                            next_value_function_list[state] += self.policy[state][action] * p * (reward + self.gamma * self.value_function_list[next_state])
                else:
                    continue
                max_diff = max(max_diff, abs(next_value_function_list[state] - self.value_function_list[state]))
            self.value_function_list = next_value_function_list
            if max_diff < self.theta:
                break
            cnt += 1
        print(f"策略评估进行{cnt}轮后完成")

    def policy_improvement(self) -> None:
        for state in self.env.P:
            if state not in self.end_states:
                action_value_function_list = np.zeros(len(self.env.P[state]))
                for action in self.env.P[state]:
                    for p, next_state, reward, terminated in self.env.P[state][action]:
                        action_value_function_list[action] += p * (reward + self.gamma * self.value_function_list[next_state])
                self.policy[state] = self.__get_state_policy(action_value_function_list)
            else:
                continue
        print("策略提升完成")

    def iteration(self) -> None:
        while True:
            self.policy_evaluation()
            old_policy = copy.deepcopy(self.policy)
            self.policy_improvement()
            if old_policy == self.policy:
                break

