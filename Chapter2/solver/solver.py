import numpy as np

from Chapter2.bandit.bandit import Bandit
import abc

class Solver(abc.ABC):
    def __init__(self, bandit: Bandit) -> None:
        self.bandit = bandit
        self.current_r = 0
        self.count_list = np.zeros(bandit.n, dtype=np.int64)
        self.action_list = list()
        self.r_list = list()

    def upgrade_r(self, k: int) -> None:
        self.current_r += self.bandit.best_probability - self.bandit.probability_list[k]
        self.r_list.append(self.current_r)

    @abc.abstractmethod
    def run_one_step(self) -> int:
        pass

    def run(self, t: int) -> None:
        for _ in range(t):
            k = self.run_one_step()
            self.upgrade_r(k)
            self.count_list[k] += 1
            self.action_list.append(k)
