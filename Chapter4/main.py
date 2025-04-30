import gymnasium as gym
import numpy as np

from Chapter4.env.my_cliff_walking_env import MyCliffWalkingEnv
from Chapter4.iteration.policy_iteration import PolicyIteration


def my_cliff_walking(theta: float = 0.001, gamma: float = 0.9) -> None:
    env = MyCliffWalkingEnv()
    policy_iter = PolicyIteration(env, theta, gamma)
    policy_iter.iteration()

    print(np.array(policy_iter.value_function_list))
    print(policy_iter.policy)

def gym_cliff_walking(theta: float = 0.001, gamma: float = 0.9) -> None:
    env = gym.make("CliffWalking-v0", render_mode="human").unwrapped
    policy_iter = PolicyIteration(env, theta, gamma)
    policy_iter.iteration()

    print(np.array(policy_iter.value_function_list))
    print(policy_iter.policy)

if __name__ == "__main__":
    my_cliff_walking()
    gym_cliff_walking()