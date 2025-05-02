import gymnasium as gym
from gymnasium.core import Env
from Chapter4.env.my_cliff_walking_env import MyCliffWalkingEnv
from Chapter4.iteration.base_iteration import BaseIteration
from Chapter4.iteration.policy_iteration import PolicyIteration
from Chapter4.iteration.value_iteration import ValueIteration

def gym_visualize(env: Env, iteration: BaseIteration) -> None:
    observation, info = env.reset()

    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = iteration.predict_action(observation)  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
    env.close()

def my_cliff_walking(theta: float = 0.001, gamma: float = 0.9) -> None:
    env = MyCliffWalkingEnv()
    policy_iter = PolicyIteration(env, theta, gamma)
    policy_iter.iteration()

    policy_iter.output_v_policy()

    value_iter = ValueIteration(env, theta, gamma)
    value_iter.iteration()

    policy_iter.output_v_policy()

def gym_cliff_walking(theta: float = 0.001, gamma: float = 0.9) -> None:
    env = gym.make("CliffWalking-v0", render_mode="human").unwrapped
    policy_iter = PolicyIteration(env, theta, gamma)
    policy_iter.iteration()

    policy_iter.output_v_policy()

    value_iter = ValueIteration(env, theta, gamma)
    value_iter.iteration()

    policy_iter.output_v_policy()

    gym_visualize(env, policy_iter)

def gym_frozen_lake() -> None:
    pass

if __name__ == "__main__":
    my_cliff_walking()

    gym_cliff_walking()

    gym_frozen_lake()