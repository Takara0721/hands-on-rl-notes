import numpy as np
from Chapter2.bandit.bandit import Bandit
from .solver import Solver


class ThompsonSampling(Solver):
    def __init__(self, bandit: Bandit):
        super().__init__(bandit)
        self.success_count_list = np.ones(self.bandit.n)
        self.failed_count_list = np.ones(self.bandit.n)

    def run_one_step(self) -> int:
        sample = np.random.beta(self.success_count_list, self.failed_count_list)
        k = int(np.argmax(sample))
        r = self.bandit.step(k)
        if r > 0:
            self.success_count_list[k] += 1
        else:
            self.failed_count_list[k] += 1
        return k