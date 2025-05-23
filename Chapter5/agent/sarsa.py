from gymnasium.core import ObsType, ActType, Env
from tqdm import tqdm
from Chapter5.agent.rl_agent_base import RLAgentBase
import numpy as np


class Sarsa(RLAgentBase):
    def __init__(self, env: Env, alpha: float, gamma: float, epsilon: float):
        super().__init__(env, alpha, gamma, epsilon)

    def take_action(self, state: ObsType) -> ActType:
        if np.random.random() < self.epsilon:
            # action = self.env.action_space.sample()
            action = np.random.randint(self.env.nA)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def update(
            self,
            state: ObsType,
            action: ActType,
            reward: float,
            next_state: ObsType,
            next_action: ActType
    ) -> None:
        td_error = reward + self.gamma * self.Q_table[next_state][next_action] - self.Q_table[state][action]
        self.Q_table[state][action] += self.alpha * td_error

    def output_policy(self) -> None:
        if hasattr(self.env, 'shape'):
            map_shape = self.env.shape
        else:
            map_shape = (self.env.nrow, self.env.ncol)

        action_str_list = ['↑', '→', '↓', '←']
        policy_str_list = []
        for state in range(self.env.nS):
            tmp = ''
            state_policy = self.best_action(state)
            for action, prob in enumerate(state_policy):
                if prob:
                    tmp += action_str_list[action]
            policy_str_list.append(tmp)

        policy_map = np.array(policy_str_list).reshape(map_shape)
        padding_width = max(len(str(s)) for s in policy_map.flat) + 1

        with np.printoptions(linewidth=np.inf, formatter={'str_kind': lambda x: f'{x:<{padding_width}}'}):
            print(policy_map)

    def train(self, num_episodes: int, episode_batch_num: int) -> list[float]:
        episodes_per_batch = num_episodes // episode_batch_num
        for i in range(episode_batch_num):
            total = episodes_per_batch if i != episode_batch_num - 1 else num_episodes - episodes_per_batch * i
            with tqdm(total=total, desc=f"Iteration {i}") as pbar:
                for episode_i in range(total):
                    episode_return = 0
                    state, info = self.env.reset()
                    action = self.take_action(state)
                    done = False
                    while not done:
                        next_state, reward, done, truncated, info = self.env.step(action)
                        next_action = self.take_action(next_state)
                        episode_return += reward
                        self.update(state, action, reward, next_state, next_action)
                        state, action = next_state, next_action
                    self.return_list.append(episode_return)
                    if (episode_i + 1) % 10 == 0 or episode_i == total - 1:
                        pbar.set_postfix({
                            "episode": f"{num_episodes / 10 * i + episode_i + 1}",
                            "return": f"{np.mean(self.return_list[-10:]):.3f}"
                        })
                    pbar.update(1)