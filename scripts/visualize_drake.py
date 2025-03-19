import gymnasium
import PyFlyt.gym_envs # noqa
from PyFlyt.gym_envs import FlattenWaypointEnv # noqa

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from pydrake.all import StartMeshcat
from cs175.quadrotor_env import QuadrotorEnv, QuadrotorPlant

# Quadrotor LQR example notebook:
# https://deepnote.com/workspace/Underactuated-2ed1518a-973b-4145-bd62-1768b49956a8/project/096cffe7-416e-4d51-a471-5fd526ec8fab/notebook/quadrotor-ac8fa093ca5a4894a618ada316f2750b

meshcat = StartMeshcat()

writer = SummaryWriter()



# env = make_vec_env(lambda: FlattenWaypointEnv(gymnasium.make("PyFlyt/QuadX-Waypoints-v3", render_mode='human'), context_length=1), n_envs=1)

env = QuadrotorEnv.build(meshcat, np.zeros(3), np.zeros(3), realtime_rate=1)


model = PPO.load("drake_quadrotor_stabilize_working")
rewards = []
for episode in range(10):
    observation, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = truncated or terminated
    
    rewards.append(episode_reward)

env.close()

print(rewards)

input("Press Enter to close the window and end the program...")