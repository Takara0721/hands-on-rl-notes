import abc

class Bandit(abc.ABC):
    def __init__(self, n: int) -> None:
        self.probability_list = None
        self.best_index = -1
        self.best_probability = 0
        self.n = n

    @abc.abstractmethod
    def step(self, k: int) -> float:
        pass