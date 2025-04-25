from .base_states import BaseStates
from .state import State

class StudentStates(BaseStates):
    Facebook = State(0, -1)
    Class1 = State(1, -2)
    Class2 = State(2, -2)
    Class3 = State(3, -2)
    Pub = State(4, 1)
    Pass = State(5, 1)
    Sleep = State(6, 0)
