import numpy as np
from .base_iteration import BaseIteration
from gymnasium.core import Env
from Chapter4.env.my_cliff_walking_env import MyCliffWalkingEnv

class ValueIteration(BaseIteration):
    def __init__(self, env: Env | MyCliffWalkingEnv, theta: float, gamma: float) -> None:
        super().__init__(env, theta, gamma)

    def iteration(self) -> None:
        cnt = 0
        while True:
            next_value_function_list = np.zeros(len(self.env.P))
            for state in self.env.P:
                if state not in self.end_states:
                    action_value_function_list = np.zeros(len(self.env.P[state]))
                    for action in self.env.P[state]:
                        for p, next_state, reward, terminated in self.env.P[state][action]:
                            action_value_function_list[action] += p * (reward + self.gamma * self.value_function_list[next_state])
                else:
                    continue
                next_value_function_list[state] = np.max(action_value_function_list)
            max_diff = np.max(np.abs(next_value_function_list - self.value_function_list))
            self.value_function_list = next_value_function_list
            if max_diff < self.theta:
                break
            cnt += 1

        print(f"价值迭代一共进行{cnt}轮")
        self._from_v_get_policy()