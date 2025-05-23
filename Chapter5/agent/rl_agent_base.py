import abc
from abc import abstractmethod
import numpy as np
from gymnasium.core import ActType, ObsType, Env
from matplotlib import pyplot as plt


class RLAgentBase(abc.ABC):
    def __init__(self, env: Env, alpha: float, gamma: float, epsilon: float) -> None:
        self.env = env.unwrapped
        self.Q_table = np.zeros((self.env.nS, self.env.nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_list = []

    def best_action(self, state: ObsType) -> ActType:
        np_list = self.Q_table[state]
        max_condition = (np_list == np.max(np_list))
        max_num = np.sum(max_condition)
        state_policy = max_condition.astype(float)
        state_policy /= max_num
        return state_policy.tolist()

    def plot_returns(self) -> None:
        if not self.return_list:
            raise Exception("模型未进行训练，无采样数据！")
        episodes_list = list(range(len(self.return_list)))
        plt.plot(episodes_list, self.return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('Sarsa on {}'.format('Cliff Walking'))
        plt.show()

    @abstractmethod
    def take_action(self, state: ObsType) -> ActType:
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def train(self, num_episodes: int, episode_batch_num: int) -> None:
        pass