from Chapter3.action.base_action import BaseActionEnum
from Chapter3.state.base_state_enum import BaseStateEnum

class SAPair:
    def __init__(self, state: BaseStateEnum, action: BaseActionEnum):
        self.state = state
        self.action = action

    def __eq__(self, other: object) -> bool | None:
        if isinstance(other, SAPair):
            return (
                    self.state == other.state
                    and self.action == other.action
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.state, self.action))

    def __repr__(self) -> str:
        return f"SAPair(state={self.state.name}, action={self.action.name})"