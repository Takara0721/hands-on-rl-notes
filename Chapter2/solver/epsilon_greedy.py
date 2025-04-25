from .solver import Solver
from Chapter2.bandit.bandit import Bandit
import numpy as np

class EpsilonGreedy(Solver):
    def __init__(
            self,
            bandit: Bandit,
            epsilon: float = 0.01,
            init_probability: float = 1.0
    ) -> None:
        super().__init__(bandit)
        self.epsilon = epsilon
        self.q_list = np.array([init_probability] * self.bandit.n)

    def run_one_step(self) -> int:
        if np.random.random() > self.epsilon:
            k = np.argmax(self.q_list)
        else:
            k = np.random.randint(self.bandit.n)
        r = self.bandit.step(k)
        self.q_list[k] += (r - self.q_list[k]) / (self.count_list[k] + 1)
        return k