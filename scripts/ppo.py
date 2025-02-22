import numpy as np
from tqdm import tqdm
from utils import load_config, intialize_loggin, create_vectorized_env, train_ppo
from pydrake.all import StartMeshcat

# Quadrotor LQR example notebook:
# https://deepnote.com/workspace/Underactuated-2ed1518a-973b-4145-bd62-1768b49956a8/project/096cffe7-416e-4d51-a471-5fd526ec8fab/notebook/quadrotor-ac8fa093ca5a4894a618ada316f2750b

meshcat = StartMeshcat()

config = load_config()["ppo"]

writer, logdir = initalize_logging("drake_quadrotor_ppo_default_straighthover")

target_position = np.array([0, 0, 1])
target_velocity = np.array([0, 0, 0])

env = create_vectorized_env("QuadrotorEnv", config["num_envs"], meshcat, target_position, target_velocity)

train_ppo(env, config, logdir)

input("Press Enter to close the window and end the program...")
