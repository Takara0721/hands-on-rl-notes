from Chapter3.action.base_actions import BaseActions
from Chapter3.state.base_states import BaseStates

class Reward:
    def __init__(self, state: BaseStates, action: BaseActions, value: float):
        self.state = state
        self.action = action
        self.value = value

    def __eq__(self, other: object) -> bool | None:
        if isinstance(other, Reward):
            return (
                    self.state.index == other.state.index
                    and self.state.reward == other.state.reward
                    and self.action == other.action
            )
        return NotImplemented

    def __repr__(self) -> str:
        return f"Reward(state={self.state.name}, action={self.action.name})"