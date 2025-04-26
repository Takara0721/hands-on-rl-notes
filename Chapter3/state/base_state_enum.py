import enum

class BaseStateEnum(enum.Enum):
    @property
    def index(self) -> int:
        return self.value.index

    @property
    def reward(self) -> float | None:
        return self.value.reward