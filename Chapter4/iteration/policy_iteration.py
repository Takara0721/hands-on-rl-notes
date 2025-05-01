import copy
import numpy as np
from .base_iteration import BaseIteration
from gymnasium.core import Env
from Chapter4.env.my_cliff_walking_env import MyCliffWalkingEnv

class PolicyIteration(BaseIteration):
    def __init__(self, env: Env | MyCliffWalkingEnv, theta: float, gamma: float) -> None:
        super().__init__(env, theta, gamma)

    def policy_evaluation(self) -> None:
        cnt = 1
        while True:
            next_value_function_list = np.zeros(len(self.env.P))
            for state in self.env.P:
                if state not in self.end_states:
                    for action in self.env.P[state]:
                        for p, next_state, reward, terminated in self.env.P[state][action]:
                            next_value_function_list[state] += self.policy[state][action] * p * (reward + self.gamma * self.value_function_list[next_state])
                else:
                    continue
            max_diff = np.max(np.abs(next_value_function_list - self.value_function_list))
            self.value_function_list = next_value_function_list
            if max_diff < self.theta:
                break
            cnt += 1
        print(f"策略评估进行{cnt}轮后完成")

    def policy_improvement(self) -> None:
        self._from_v_get_policy()
        print("策略提升完成")

    def iteration(self) -> None:
        while True:
            self.policy_evaluation()
            old_policy = copy.deepcopy(self.policy)
            self.policy_improvement()
            if old_policy == self.policy:
                break
