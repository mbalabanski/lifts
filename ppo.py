import gymnasium as gym

import lifts

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("lifts/QuadRotor-v0", n_envs=4)

model = PPO("MultiInputPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

env = gym.make("lifts/QuadRotor-v0", render_mode="human")

while True:

    obs, _ = env.reset()

    for t in range(1, 10000):

        # select action from policy
        action, _ = model.predict(np.array(obs))

        action = np.array(action).flatten()

        # take the action
        obs, reward, done, _, _ = env.step(action)

        if done:
            break