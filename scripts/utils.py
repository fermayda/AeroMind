import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pydrake.all import StartMeshcat
from cs175.quadrotor_env import QuadrotorEnv

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_env(env_name, meshcat, target_position, target_velocity):
    env = QuadrotorEnv.build(meshcat, target_position, target_velocity)
    return Monitor(env)


def create_vectorized_env(env_name, num_envs, meshcat, target_position, target_velocity):
    def make_single_env():
        return lambda: make_env(env_name, meshcat, target_position, target_velocity)

    return SubprocVecEnv([make_single_env() for _ in range(num_envs)])


def initialize_logging(experiment_name):
    logdir = f"runs/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    return SummaryWriter(logdir), logdir


def train_ppo(env, config, logdir):
        model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logdir,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        device=config["device"],
    )
    model.learn(total_timesteps=config["total_timesteps"])
    model.save("ppo_quadrotor_stabilize")
    return model