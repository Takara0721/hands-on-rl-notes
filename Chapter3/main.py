from Chapter3.base.state_action_pair import SAPair
from action.my_action_enum import MyActionEnum
from action.base_action import ActionDetail
from Chapter3.transition.state_transition import StateTransition
from state.my_state_enum import MyStateEnum, MyMDPStateEnum
from markov.markov_reward_process import MarkovRewardProcess
from markov.markov_decision_process import MarkovDecisionProcess

def markov_reward_process(
        _markov_chain: dict[StateTransition, float],
        gamma: float = 0.5
) -> None:
    mrp = MarkovRewardProcess(MyStateEnum, _markov_chain, gamma)
    episode = [MyStateEnum.S1, MyStateEnum.S2, MyStateEnum.S3, MyStateEnum.S6]
    G = mrp.compute_return(episode)

    chain_str = "->".join(map(lambda x: x.name, episode))

    print(f"序列 {chain_str} 计算得到回报为：{G}")

    value_function_list = mrp.compute_value_function_list()

    print(f"MRP中每个状态价值分别为\n{value_function_list}")

def markov_decision_process(
        _state_action_details: dict[SAPair, ActionDetail],
        _markov_chain: dict[StateTransition, float],
        gamma: float = 0.5
) -> None:
    mdp = MarkovDecisionProcess(MyMDPStateEnum, MyActionEnum, _state_action_details, gamma, _markov_chain)
    episode = [
        SAPair(MyMDPStateEnum.S1, MyActionEnum.GoS2),
        SAPair(MyMDPStateEnum.S2, MyActionEnum.GoS3),
        SAPair(MyMDPStateEnum.S3, MyActionEnum.GoS5)
    ]
    G = mdp.compute_return(episode)

    chain_str = "->".join(map(lambda x: f"{x.state.name}-{x.action.name}", episode))

    print(f"序列 {chain_str} 计算得到回报为：{G}")

    state_value_function_list = mdp.compute_state_value_function_list()

    print(f"MDP中每个状态价值分别为\n{state_value_function_list}")

    action_value_function = mdp.compute_action_value_function(MyMDPStateEnum.S4, MyActionEnum.RandomGo)

    print(f"(S4, 概率前往)的动作价值函数为：{action_value_function}")

    evaluation_svf_list = mdp.monte_carlo_policy_evaluation_v(20, 1000)

    print(f"使用蒙特卡洛方法计算MDP的状态价值为\n{evaluation_svf_list}")

    rho1 = mdp.occupancy_evaluation(MyMDPStateEnum.S4, MyActionEnum.RandomGo, 20, 1000)

    print(f"随机策略下，(S4, 概率前往)的占用度量估计为：{rho1}")

    policy = {
        SAPair(MyMDPStateEnum.S1, MyActionEnum.KeepS1): 0.6,
        SAPair(MyMDPStateEnum.S1, MyActionEnum.GoS2): 0.4,
        SAPair(MyMDPStateEnum.S2, MyActionEnum.GoS1): 0.3,
        SAPair(MyMDPStateEnum.S2, MyActionEnum.GoS3): 0.7,
        SAPair(MyMDPStateEnum.S3, MyActionEnum.GoS4): 0.5,
        SAPair(MyMDPStateEnum.S3, MyActionEnum.GoS5): 0.5,
        SAPair(MyMDPStateEnum.S4, MyActionEnum.GoS5): 0.1,
        SAPair(MyMDPStateEnum.S4, MyActionEnum.RandomGo): 0.9
    }
    mdp.change_policy(policy)

    rho2 = mdp.occupancy_evaluation(MyMDPStateEnum.S4, MyActionEnum.RandomGo, 20, 1000)

    print(f"书中给出的策略下，(S4, 概率前往)的占用度量估计为：{rho2}")


if __name__ == '__main__':
    mrp_markov_chain = {
        StateTransition(MyStateEnum.S1, MyStateEnum.S1): 0.9,
        StateTransition(MyStateEnum.S1, MyStateEnum.S2): 0.1,
        StateTransition(MyStateEnum.S2, MyStateEnum.S1): 0.5,
        StateTransition(MyStateEnum.S2, MyStateEnum.S3): 0.5,
        StateTransition(MyStateEnum.S3, MyStateEnum.S4): 0.6,
        StateTransition(MyStateEnum.S3, MyStateEnum.S6): 0.4,
        StateTransition(MyStateEnum.S4, MyStateEnum.S5): 0.3,
        StateTransition(MyStateEnum.S4, MyStateEnum.S6): 0.7,
        StateTransition(MyStateEnum.S5, MyStateEnum.S2): 0.2,
        StateTransition(MyStateEnum.S5, MyStateEnum.S3): 0.3,
        StateTransition(MyStateEnum.S5, MyStateEnum.S4): 0.5
    }
    markov_reward_process(mrp_markov_chain)

    mdp_markov_chain = {
        StateTransition(MyMDPStateEnum.S1, MyMDPStateEnum.S1, MyActionEnum.KeepS1): 1.0,
        StateTransition(MyMDPStateEnum.S1, MyMDPStateEnum.S2, MyActionEnum.GoS2): 1.0,
        StateTransition(MyMDPStateEnum.S2, MyMDPStateEnum.S1, MyActionEnum.GoS1): 1.0,
        StateTransition(MyMDPStateEnum.S2, MyMDPStateEnum.S3, MyActionEnum.GoS3): 1.0,
        StateTransition(MyMDPStateEnum.S3, MyMDPStateEnum.S4, MyActionEnum.GoS4): 1.0,
        StateTransition(MyMDPStateEnum.S3, MyMDPStateEnum.S5, MyActionEnum.GoS5): 1.0,
        StateTransition(MyMDPStateEnum.S4, MyMDPStateEnum.S5, MyActionEnum.GoS5): 1.0,
        StateTransition(MyMDPStateEnum.S4, MyMDPStateEnum.S2, MyActionEnum.RandomGo): 0.2,
        StateTransition(MyMDPStateEnum.S4, MyMDPStateEnum.S3, MyActionEnum.RandomGo): 0.4,
        StateTransition(MyMDPStateEnum.S4, MyMDPStateEnum.S4, MyActionEnum.RandomGo): 0.4
    }
    state_action_details = {
        SAPair(MyMDPStateEnum.S1, MyActionEnum.KeepS1): ActionDetail(-1, 0.5),
        SAPair(MyMDPStateEnum.S1, MyActionEnum.GoS2): ActionDetail(0, 0.5),
        SAPair(MyMDPStateEnum.S2, MyActionEnum.GoS1): ActionDetail(-1, 0.5),
        SAPair(MyMDPStateEnum.S2, MyActionEnum.GoS3): ActionDetail(-2, 0.5),
        SAPair(MyMDPStateEnum.S3, MyActionEnum.GoS4): ActionDetail(-2, 0.5),
        SAPair(MyMDPStateEnum.S3, MyActionEnum.GoS5): ActionDetail(0, 0.5),
        SAPair(MyMDPStateEnum.S4, MyActionEnum.GoS5): ActionDetail(10, 0.5),
        SAPair(MyMDPStateEnum.S4, MyActionEnum.RandomGo): ActionDetail(1, 0.5)
    }
    markov_decision_process(state_action_details, mdp_markov_chain)