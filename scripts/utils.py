import gymnasium as gym
import numpy as np
import yaml
import optuna 
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
    checkpoint_dr = "checkpoints"
    os.makedirs(checkpoint_dr, exist_ok=True)

    for step in range(0, config["total_timesteps"], 100000):
        model.learn(total_timesteps=100000, reset_num_timesteps=False)
        check_pth = os.path.join(checkpoint_dr, f"ppo_checkpoints_{step+100000}.zip")
        model.save(checkpoint_path)
        print(f"checkpoint saved at: {checkpoint_path}")

    model.save("ppo_quadrotor_stabilize")
    return model

def objective(trial, env, logdir):
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
        "n_steps": trial.suggest_int("n_steps", 512, 4096, step=512),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_uniform("gamma", 0.9, 0.999),
    }

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=logdir,
        **params,
    )
    model.learn(total_timesteps=100000)
    mean_reward = evaluate_model(model, env)
    return mean_reward


def evaluate_model(model, env, n_eval_episodes=5):
    total_reward = 0
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / n_eval_episodes


def tune_hyperparameters(env, logdir, n_trials=20):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, env, logdir), n_trials=n_trials)

    best_params = study.best_params
    print(f"best hyperparameters: {best_params}")
    return best_params