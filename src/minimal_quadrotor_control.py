import numpy as np
from pydrake.all import (
    DiagramBuilder, SceneGraph, Simulator, System, Context,
    StartMeshcat, MeshcatVisualizer 
)
from pydrake.examples import QuadrotorPlant, QuadrotorGeometry, StabilizingLQRController
from pydrake.gym._drake_gym_env import DrakeGymEnv
import gymnasium as gym
from tqdm import tqdm

# Quadrotor LQR example notebook:
# https://deepnote.com/workspace/Underactuated-2ed1518a-973b-4145-bd62-1768b49956a8/project/096cffe7-416e-4d51-a471-5fd526ec8fab/notebook/quadrotor-ac8fa093ca5a4894a618ada316f2750b

meshcat = StartMeshcat()

def quadrotor_example():
    builder = DiagramBuilder()

    plant = builder.AddSystem(QuadrotorPlant())

    # controller = builder.AddSystem(StabilizingLQRController(plant, [0, 0, 1]))
    # builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    # builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

    PLANT_INPUT_NAME = "plant_input"
    PLANT_OUTPUT_NAME = "state"
    plant_input_port = builder.ExportInput(plant.get_input_port(0), PLANT_INPUT_NAME)
    plant_output_port = builder.ExportOutput(plant.get_output_port(0), PLANT_OUTPUT_NAME)


    # Set up visualization in MeshCat
    scene_graph = builder.AddSystem(SceneGraph())
    QuadrotorGeometry.AddToBuilder(builder, plant.get_output_port(0), scene_graph)
    meshcat.Delete()
    meshcat.ResetRenderMode()
    meshcat.SetProperty("/Background", "visible", False)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    # end setup for visualization

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    target_position = np.array([0, 0, 1])
    target_velocity = np.array([0, 0, 0])

    def reward_fn(system: System, context: Context):
        state = system.get_output_port(0).Eval(context) 
        # print(state.shape, state[:3])
        pos_err = np.linalg.norm(state[:3] - target_position)
        vel_err = np.linalg.norm(state[3:6] - target_velocity)
        return -pos_err - vel_err

    gym_env = DrakeGymEnv(
        simulator, 
        0.001, 
        gym.spaces.Box(low=0, high=100, shape=(4,)),
        gym.spaces.Box(low=-100, high=100, shape=(12,)),
        reward_fn,
        PLANT_INPUT_NAME,
        PLANT_OUTPUT_NAME,
    )

    dummy_plant = QuadrotorPlant()
    controller = StabilizingLQRController(dummy_plant, target_position)
    controller_context = controller.CreateDefaultContext()

    for _ in range(5):
        print("Starting episode")
        gym_env.reset()
        action = np.zeros(4)
        for _ in tqdm(range(1000)):
            observation, reward, terminated, truncated, info = gym_env.step(action)
            if terminated:
                break
            controller.get_input_port(0).FixValue(controller_context, observation)
            action = controller.get_output_port(0).Eval(controller_context)
        input("Press Enter to start the next episode...")


quadrotor_example()

input("Press Enter to close the window and end the program...")