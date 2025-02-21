import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pydrake.all import StartMeshcat
from cs175.quadrotor_env import QuadrotorEnv, QuadrotorPlant
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from datetime import datetime

# Quadrotor LQR example notebook:
# https://deepnote.com/workspace/Underactuated-2ed1518a-973b-4145-bd62-1768b49956a8/project/096cffe7-416e-4d51-a471-5fd526ec8fab/notebook/quadrotor-ac8fa093ca5a4894a618ada316f2750b

meshcat = StartMeshcat()

writer = SummaryWriter()

experiment_name='drake_quadrotor_ppo_default_straighthover'
experiment_logdir = f"runs/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

def quadrotor_example():

    target_position = np.array([0, 0, 1])
    target_velocity = np.array([0, 0, 0])

    env = QuadrotorEnv.build(meshcat, target_position, target_velocity)
    env = Monitor(env)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=experiment_logdir)
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_quadrotor_stabilize")

quadrotor_example()

input("Press Enter to close the window and end the program...")
