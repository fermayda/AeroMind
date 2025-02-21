import gymnasium
import PyFlyt.gym_envs # noqa

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from cs175.tensorboard_video_recorder import TensorboardVideoRecorder
from datetime import datetime

# Quadrotor LQR example notebook:
# https://deepnote.com/workspace/Underactuated-2ed1518a-973b-4145-bd62-1768b49956a8/project/096cffe7-416e-4d51-a471-5fd526ec8fab/notebook/quadrotor-ac8fa093ca5a4894a618ada316f2750b

writer = SummaryWriter()

experiment_name='pyflyt_quadrotor_ppo_default_straighthover'
experiment_logdir = f"runs/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

env = gymnasium.make("PyFlyt/QuadX-Hover-v3", render_mode="rgb_array")
# Define a trigger function (e.g., record a video every 20,000 steps)
env = Monitor(env)
video_trigger = lambda step: step % 2e4 == 0
# Wrap the environment in a monitor that records videos to Tensorboard
env = TensorboardVideoRecorder(env=env,
                                video_trigger=video_trigger,
                                video_length=500,
                                fps=30,
                                tb_log_dir=experiment_logdir)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=experiment_logdir, device='cpu')
model.learn(total_timesteps=1_000_000)
model.save("ppo_quadrotor_stabilize")
