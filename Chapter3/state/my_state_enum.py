from .base_state_enum import BaseStateEnum
from .state import State

class MyStateEnum(BaseStateEnum):
    S1 = State(0, -1)
    S2 = State(1, -2)
    S3 = State(2, -2)
    S4 = State(3, 10)
    S5 = State(4, 1)
    S6 = State(5, 0)

class MyMDPStateEnum(BaseStateEnum):
    S1 = State(0)
    S2 = State(1)
    S3 = State(2)
    S4 = State(3)
    S5 = State(4)