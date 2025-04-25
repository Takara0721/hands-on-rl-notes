from transition.state_transition import StateTransition
from state.base_states import BaseStates
from state.my_states import MyStates
from markov.markov_reward_process import MarkovRewardProcess

def student_markov_reward_process(
        states: type[BaseStates],
        _markov_chain: list[StateTransition],
        gamma: float = 0.5
) -> None:
    markov_reward_process = MarkovRewardProcess(states, _markov_chain, gamma)
    chain = [MyStates.S1, MyStates.S2, MyStates.S3, MyStates.S6]
    G = markov_reward_process.compute_return(chain)
    print(f"根据本序列计算得到回报为：{G}")

if __name__ == '__main__':
    markov_chain = [
        StateTransition(MyStates.S1, MyStates.S1, 0.9),
        StateTransition(MyStates.S1, MyStates.S2, 0.1),
        StateTransition(MyStates.S2, MyStates.S1, 0.5),
        StateTransition(MyStates.S2, MyStates.S3, 0.5),
        StateTransition(MyStates.S3, MyStates.S4, 0.6),
        StateTransition(MyStates.S3, MyStates.S6, 0.4),
        StateTransition(MyStates.S4, MyStates.S5, 0.3),
        StateTransition(MyStates.S4, MyStates.S6, 0.7),
        StateTransition(MyStates.S5, MyStates.S2, 0.2),
        StateTransition(MyStates.S5, MyStates.S3, 0.3),
        StateTransition(MyStates.S5, MyStates.S4, 0.5)
    ]
    student_markov_reward_process(MyStates, markov_chain)