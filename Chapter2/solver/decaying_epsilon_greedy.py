from typing import Callable
import numpy as np
from Chapter2.bandit.bandit import Bandit
from .solver import Solver


class DecayingEpsilonGreedy(Solver):
    def __init__(
            self,
            bandit: Bandit,
            decaying_epsilon: Callable[[int], int] = lambda t : 1 / t,
            init_probability: float = 1.0
    ) -> None:
        super().__init__(bandit)
        self.t = 0
        self.decaying_epsilon = decaying_epsilon
        self.q_list = np.array([init_probability] * self.bandit.n)

    def run_one_step(self) -> int:
        self.t += 1
        if np.random.random() > self.decaying_epsilon(self.t):
            k = np.argmax(self.q_list)
        else:
            k = np.random.randint(self.bandit.n)
        r = self.bandit.step(k)
        self.q_list[k] += (r - self.q_list[k]) / (self.count_list[k] + 1)
        return k

