import abc

class Markov(abc.ABC):
    def __init__(self, state_list: list[str], reward_list: list[float], gamma: float) -> None:
        self.state_list = state_list
        self.transition_matrix = None
        self.reward_list = reward_list
        self.gamma = gamma

    @abc.abstractmethod
    def compute_return(self, start: int, chain: list):
        pass
