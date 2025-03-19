import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pydrake.all import StartMeshcat
from pydrake.examples import StabilizingLQRController
from cs175.quadrotor_env import QuadrotorEnv, QuadrotorPlant

# Quadrotor LQR example notebook:
# https://deepnote.com/workspace/Underactuated-2ed1518a-973b-4145-bd62-1768b49956a8/project/096cffe7-416e-4d51-a471-5fd526ec8fab/notebook/quadrotor-ac8fa093ca5a4894a618ada316f2750b

meshcat = StartMeshcat()

writer = SummaryWriter()

def quadrotor_example():

    target_position = np.array([0, 0, 0])
    target_velocity = np.array([0, 0, 0])

    env = QuadrotorEnv.build(None, target_position, target_velocity)

    dummy_plant = QuadrotorPlant()
    controller = StabilizingLQRController(dummy_plant, target_position)
    controller_context = controller.CreateDefaultContext()

    actions = []
    states = []
    for episode_no in range(5):
        print("Starting episode")
        cumulative_reward = 0
        env.reset()
        action = np.zeros(4)
        for _ in tqdm(range(10_000)):
            observation, reward, terminated, truncated, info = env.step(action)
            states.append(observation)
            if terminated or truncated:
                break
            controller.get_input_port(0).FixValue(controller_context, observation)
            action = controller.get_output_port(0).Eval(controller_context)
            # action = np.random.random(4)
            # action = np.zeros(4)
            actions.append(action)
            cumulative_reward += reward
        writer.add_scalar("reward", cumulative_reward, episode_no)
        print(cumulative_reward)

    actions = np.array(actions)
    states = np.array(states)
    print(actions.min())
    print(actions.max())
    print(states.min())
    print(states.max())

quadrotor_example()

input("Press Enter to close the window and end the program...")