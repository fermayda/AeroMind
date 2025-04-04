import gymnasium
import PyFlyt.gym_envs # noqa
from PyFlyt.gym_envs import FlattenWaypointEnv # noqa

from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from cs175.tensorboard_video_recorder import TensorboardVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime

# Quadrotor LQR example notebook:
# https://deepnote.com/workspace/Underactuated-2ed1518a-973b-4145-bd62-1768b49956a8/project/096cffe7-416e-4d51-a471-5fd526ec8fab/notebook/quadrotor-ac8fa093ca5a4894a618ada316f2750b

writer = SummaryWriter()

experiment_name='pyflyt_quadrotor_ppo_default_straighthover_unvectorized'
experiment_logdir = f"runs/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def make_env():
    env = gymnasium.make("PyFlyt/QuadX-Waypoints-v3", render_mode='rgb_array', flight_mode=-1, agent_hz=100)
    return env
env = make_vec_env(make_env, n_envs=1)
# Define a trigger function (e.g., record a video every 20,000 steps)

def video_trigger(step):
    return step % 2e4 == 0
env = TensorboardVideoRecorder(env=env,
                                video_trigger=video_trigger,
                                video_length=500,
                                fps=30,
                                tb_log_dir=experiment_logdir)

model = PPO("MlpPolicy", env, tensorboard_log=experiment_logdir, device='cpu')
model.learn(total_timesteps=1_000_000)
model.save("ppo_quadrotor_stabilize_unvectorized")
