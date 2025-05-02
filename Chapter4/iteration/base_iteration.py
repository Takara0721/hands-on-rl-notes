import abc
import numpy as np
from gymnasium.core import Env
from Chapter4.env.my_cliff_walking_env import MyCliffWalkingEnv


class BaseIteration(abc.ABC):
    def __init_end_states(self) -> set[int]:
        end_states = set()
        for state in self.env.P:
            for action in self.env.P[state]:
                for transition_prob in self.env.P[state][action]:
                    if transition_prob[3]:
                        end_states.add(transition_prob[1])

        return end_states

    def __init__(self, env: Env | MyCliffWalkingEnv, theta: float, gamma: float) -> None:
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.end_states = self.__init_end_states()
        self.value_function_list = np.zeros(len(env.P))
        self.policy = [[1 / len(env.P[i]) for _ in range(len(env.P[i]))] for i in range(len(env.P))]

    @abc.abstractmethod
    def iteration(self) -> None:
        pass

    def __get_state_policy(
            self,
            np_list: np.ndarray
    ) -> list[float]:
        max_condition = (np_list == np.max(np_list))
        max_num = np.sum(max_condition)
        state_policy = max_condition.astype(float)
        state_policy /= max_num
        return state_policy.tolist()

    def _from_v_get_policy(self) -> None:
        for state in self.env.P:
            if state not in self.end_states:
                action_value_function_list = np.zeros(len(self.env.P[state]))
                for action in self.env.P[state]:
                    for p, next_state, reward, terminated in self.env.P[state][action]:
                        action_value_function_list[action] += p * (
                                    reward + self.gamma * self.value_function_list[next_state])
                self.policy[state] = self.__get_state_policy(action_value_function_list)
            else:
                continue

    def output_v_policy(self) -> None:
        map_shape = self.env.shape
        with np.printoptions(precision=3, linewidth=150, formatter={'float': '{: 8.3f}'.format}):
            print(self.value_function_list.reshape(map_shape))

        action_str_list = ['↑', '→', '↓', '←']
        policy_str_list = []
        for state, action_list in enumerate(self.policy):
            tmp = ''
            if state not in self.end_states:
                for action, prob in enumerate(action_list):
                    if prob:
                        tmp += action_str_list[action]
            policy_str_list.append(tmp)

        policy_map = np.array(policy_str_list).reshape(map_shape)
        padding_width = max(len(str(s)) for s in policy_map.flat) + 1

        with np.printoptions(linewidth=np.inf, formatter={'str_kind': lambda x: f'{x:<{padding_width}}'}):
            print(policy_map)

    def predict_action(self, state: int) -> int | None:
        rand = np.random.random()
        tmp = 0
        for action, prob in enumerate(self.policy[state]):
            tmp += prob
            if rand < prob:
                return action