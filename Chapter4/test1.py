import time

import gymnasium as gym
import numpy as np

from Chapter4.enum.state import StateEnum

env = gym.make(
    "CliffWalking-v0",
    render_mode="human"
)

# env = gym.make('FrozenLake-v1', render_mode="human", desc=None, map_name="4x4", is_slippery=True)

observation, info = env.reset()
time.sleep(1)
print(env.unwrapped.shape)
terminated = False
truncated = False
while not terminated and not truncated:
    action = int(input())  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print(env.unwrapped.P)
    print(observation)
    env.render()
    time.sleep(2)

env.close()