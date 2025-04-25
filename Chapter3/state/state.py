class State:
    def __init__(self, index: int, reward: float | None = None) -> None:
        self.index = index
        self.reward = reward

    def __eq__(self, other: object) -> bool | None:
        if isinstance(other, State):
            return self.index == other.index and self.reward == other.reward
        return NotImplemented

    def __repr__(self) -> str:
        return f"State(index={self.index}, reward={self.reward})"