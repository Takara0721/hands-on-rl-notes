from Chapter4.enum.action import ActionEnum
from Chapter4.enum.state import StateEnum


class MyCliffWalkingEnv:
    def __calculate_transition_prob(self, state: StateEnum, action: ActionEnum) -> list[tuple[float, int, int, bool]]:
        result = []
        cliff = {i for i in range(37, 47)}
        next_state = StateEnum(0)
        current_row = state.value // 12
        match action:
            case ActionEnum.UP:
                next_state = StateEnum(state.value - 12) if state.value - 12 >= 0 else state
            case ActionEnum.RIGHT:
                next_state = StateEnum(state.value + 1) if (state.value + 1) // 12 == current_row else state
            case ActionEnum.DOWN:
                next_state = StateEnum(state.value + 12) if state.value + 12 < 48 else state
            case ActionEnum.LEFT:
                next_state = StateEnum(state.value - 1) if (state.value - 1) // 12 == current_row else state
        reward, terminated = -1, False
        if next_state.value in cliff:
            reward, terminated = -100, True
        if next_state == StateEnum.S_3_11:
            terminated = True

        result.append((1.0, next_state.value, reward, terminated))

        return result

    def __init_P(self) -> dict[int, dict[int, list[tuple[float, int, int, bool]]]]:
        P = {
            state.value: {} for state in StateEnum
        }

        for state in StateEnum:
            for action in ActionEnum:
                P[state.value][action.value] = self.__calculate_transition_prob(state, action)

        return P

    def __init__(self) -> None:
        self.col_num = 12
        self.row_num = 4
        self.P = self.__init_P()