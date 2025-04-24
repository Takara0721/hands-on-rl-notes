import numpy as np
from bandit import Bandit
from solver import Solver


class UCB(Solver):
    def __init__(self, bandit: Bandit, coef: float = 1.0, init_probability: float = 1.0) -> None:
        super().__init__(bandit)
        self.t = 0
        self.q_list = np.array([init_probability] * self.bandit.n)
        self.coef = coef

    def run_one_step(self) -> int:
        self.t += 1
        ucb = self.q_list + self.coef * np.sqrt(np.log(self.t) / (2 * (self.count_list + 1)))
        k = int(np.argmax(ucb))
        r = self.bandit.step(k)
        self.q_list[k] += (r - self.q_list[k]) / (self.count_list[k] + 1)
        return k