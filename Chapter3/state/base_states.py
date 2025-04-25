import enum

class BaseStates(enum.Enum):
    @property
    def index(self) -> int:
        return self.value.index

    @property
    def reward(self) -> float:
        return self.value.reward