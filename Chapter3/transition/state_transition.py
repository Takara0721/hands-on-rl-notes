from Chapter3.action.base_action import BaseActionEnum
from Chapter3.state.base_state_enum import BaseStateEnum

class StateTransition:
    def __init__(self, start: BaseStateEnum, end: BaseStateEnum, action: BaseActionEnum = None) -> None:
        self.start = start
        self.end = end
        self.action = action

    def __repr__(self) -> str:
        if self.action:
            return f"StateTransition(start={self.start.name}, action={self.action.name}, end={self.end.name})"
        else:
            return f"StateTransition(start={self.start.name}, end={self.end.name})"

    def __eq__(self, other: object) -> bool | None:
        if isinstance(other, StateTransition):
            return (
                    self.start == other.start
                    and self.end == other.end
                    and self.action == other.action
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.start, self.end, self.action))
