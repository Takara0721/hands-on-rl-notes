from action.my_actions import MyActions
from reward.reward import Reward
from action.base_actions import BaseActions
from transition.state_transition import StateTransition
from state.base_states import BaseStates
from state.my_states import MyStates, MyMDPStates
from markov.markov_reward_process import MarkovRewardProcess
from markov.markov_decision_process import MarkovDecisionProcess

def markov_reward_process(
        states: type[BaseStates],
        _markov_chain: list[StateTransition],
        gamma: float = 0.5
) -> None:
    markov_reward_process = MarkovRewardProcess(states, _markov_chain, gamma)
    chain = [MyStates.S1, MyStates.S2, MyStates.S3, MyStates.S6]
    G = markov_reward_process.compute_return(chain)

    chain_str = "->".join(map(lambda x: x.name, chain))

    print(f"序列 {chain_str} 计算得到回报为：{G}")

    value_function = markov_reward_process.compute_value_function()

    print(f"MRP中每个状态价值分别为\n{value_function}")

def markov_decision_process(
        states: type[BaseStates],
        actions: type[BaseActions],
        _policy: list[tuple[BaseStates, BaseActions, float]],
        _rewards: list[Reward],
        _markov_chain: list[StateTransition],
        gamma: float = 0.5
) -> None:
    markov_decision_process = MarkovDecisionProcess(states, actions, _policy, _rewards, gamma, _markov_chain)
    chain = [
        (MyMDPStates.S1, MyActions.GoS2),
        (MyMDPStates.S2, MyActions.GoS3),
        (MyMDPStates.S3, MyActions.GoS5)
    ]
    G = markov_decision_process.compute_return(chain)

    chain_str = "->".join(map(lambda x: f"{x[0].name}-{x[1].name}", chain))

    print(f"序列 {chain_str} 计算得到回报为：{G}")

    # value_function = markov_decision_process.compute_state_value_function()
    #
    # print(f"MDP中每个状态价值分别为\n{value_function}")


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
    markov_reward_process(MyStates, markov_chain)

    markov_chain = [
        StateTransition(MyMDPStates.S1, MyStates.S1, 1.0, MyActions.KeepS1),
        StateTransition(MyMDPStates.S1, MyStates.S2, 1.0, MyActions.GoS2),
        StateTransition(MyMDPStates.S2, MyStates.S1, 1.0, MyActions.GoS1),
        StateTransition(MyMDPStates.S2, MyStates.S3, 1.0, MyActions.GoS3),
        StateTransition(MyMDPStates.S3, MyStates.S4, 1.0, MyActions.GoS4),
        StateTransition(MyMDPStates.S3, MyStates.S5, 1.0, MyActions.GoS5),
        StateTransition(MyMDPStates.S4, MyStates.S5, 1.0, MyActions.GoS5),
        StateTransition(MyMDPStates.S4, MyStates.S2, 0.2, MyActions.RandomGo),
        StateTransition(MyMDPStates.S4, MyStates.S3, 0.4, MyActions.RandomGo),
        StateTransition(MyMDPStates.S4, MyStates.S4, 0.4, MyActions.RandomGo)
    ]
    rewards = [
        Reward(MyMDPStates.S1, MyActions.KeepS1, -1),
        Reward(MyMDPStates.S1, MyActions.GoS2, 0),
        Reward(MyMDPStates.S2, MyActions.GoS1, -1),
        Reward(MyMDPStates.S2, MyActions.GoS3, -2),
        Reward(MyMDPStates.S3, MyActions.GoS4, -2),
        Reward(MyMDPStates.S3, MyActions.GoS5, 0),
        Reward(MyMDPStates.S4, MyActions.GoS5, 10),
        Reward(MyMDPStates.S4, MyActions.RandomGo, 1)
    ]
    policy = [
        (MyMDPStates.S1, MyActions.KeepS1, 0.5),
        (MyMDPStates.S1, MyActions.GoS2, 0.5),
        (MyMDPStates.S2, MyActions.GoS1, 0.5),
        (MyMDPStates.S2, MyActions.GoS3, 0.5),
        (MyMDPStates.S3, MyActions.GoS4, 0.5),
        (MyMDPStates.S3, MyActions.GoS5, 0.5),
        (MyMDPStates.S4, MyActions.GoS5, 0.5),
        (MyMDPStates.S4, MyActions.RandomGo, 0.5)
    ]
    markov_decision_process(MyMDPStates, MyActions, policy, rewards, markov_chain)