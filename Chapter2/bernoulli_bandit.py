import numpy as np
from bandit import Bandit

class BernoulliBandit(Bandit):
    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.probability_list = np.random.random(n)
        self.best_index = np.argmax(self.probability_list)
        self.best_probability = np.max(self.probability_list)

    def step(self, k: int) -> float:
        if np.random.random() >= self.probability_list[k]:
            return 0.0
        else:
            return 1.0