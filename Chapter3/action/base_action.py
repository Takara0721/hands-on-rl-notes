import enum

class BaseActionEnum(int, enum.Enum):
    pass

class ActionDetail:
    def __init__(self, reward: float, probability: float):
        self.reward = reward
        self.probability = probability

    def __repr__(self) -> str:
        return f"ActionDetail(reward={self.reward}, probability={self.probability})"