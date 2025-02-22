import numpy as np
from utils import load_config, initialize_logging, create_vectorized_env, tune_hyperparameters
from pydrake.all import StartMeshcat

meshcat = StartMeshcat()

config = load_config()["ppo"]

writer, logdir = initialize_logging("optuna_quadrotor_tuning")

target_position = np.array([0, 0, 1])
target_velocity = np.array([0, 0, 0])

env = create_vectorized_env("QuadrotorEnv", config["num_envs"], meshcat, target_position, target_velocity)

params = tune_hyperparameters(env, logdir)

import yaml
config.update(params)
with open("config.yaml", "w") as f:
    yaml.dump({"ppo": config}, f)