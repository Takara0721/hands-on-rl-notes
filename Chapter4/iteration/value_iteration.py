from .base_iteration import BaseIteration
from gymnasium.core import Env
from Chapter4.env.my_cliff_walking_env import MyCliffWalkingEnv

class ValueIteration(BaseIteration):
    def __init__(self, env: Env | MyCliffWalkingEnv, theta: float, gamma: float) -> None:
        super().__init__(env, theta, gamma)

    def iteration(self) -> None:
        pass