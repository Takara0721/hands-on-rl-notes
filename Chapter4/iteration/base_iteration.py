import abc
from gymnasium.core import Env
from Chapter4.env.my_cliff_walking_env import MyCliffWalkingEnv


class BaseIteration(abc.ABC):
    def __init__(self, env: Env | MyCliffWalkingEnv, theta: float, gamma: float) -> None:
        self.env = env
        self.theta = theta
        self.gamma = gamma

    @abc.abstractmethod
    def iteration(self) -> None:
        pass
