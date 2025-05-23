import numpy as np
from Chapter5.agent.q_learning import QLearning
from Chapter5.agent.sarsa import Sarsa
import gymnasium as gym


def gym_cliff_walking_sarsa(
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        num_episodes: int = 500,
        episode_batch_num: int = 1
) -> None:
    env = gym.make('CliffWalking-v0')

    sarsa = Sarsa(env, alpha, gamma, epsilon)

    sarsa.train(num_episodes, episode_batch_num)

    sarsa.plot_returns()

    sarsa.output_policy()

def gym_cliff_walking_q_learning(
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        num_episodes: int = 500,
        episode_batch_num: int = 1
) -> None:
    env = gym.make('CliffWalking-v0')

    q_learning = QLearning(env, alpha, gamma, epsilon)

    q_learning.train(num_episodes, episode_batch_num)

    q_learning.plot_returns()

    q_learning.output_policy()

if __name__ == "__main__":
    np.random.seed(1)
    gym_cliff_walking_sarsa()

    np.random.seed(0)
    gym_cliff_walking_q_learning()