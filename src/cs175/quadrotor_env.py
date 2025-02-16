import gymnasium as gym

from pydrake.all import (
    DiagramBuilder, SceneGraph, Simulator, System, Context,
    MeshcatVisualizer 
)
from pydrake.examples import QuadrotorPlant, QuadrotorGeometry
from pydrake.gym._drake_gym_env import DrakeGymEnv
import numpy as np

class QuadrotorEnv:
    @staticmethod
    def build(meshcat, target_position: np.ndarray, target_velocity: np.ndarray):
        builder = DiagramBuilder()

        plant = builder.AddSystem(QuadrotorPlant())


        PLANT_INPUT_NAME = "plant_input"
        PLANT_OUTPUT_NAME = "state"
        builder.ExportInput(plant.get_input_port(0), PLANT_INPUT_NAME)
        builder.ExportOutput(plant.get_output_port(0), PLANT_OUTPUT_NAME)


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
        input("Press enter once meshcat has finished initializing and you're ready")


        def reward_fn(system: System, context: Context):
            state = system.get_output_port(0).Eval(context) 
            # print(state.shape, state[:3])
            pos_err = np.linalg.norm(state[:3] - target_position)
            vel_err = np.linalg.norm(state[3:6] - target_velocity)
            return -pos_err - vel_err

        return DrakeGymEnv(
            simulator, 
            0.001, 
            gym.spaces.Box(low=0, high=100, shape=(4,)),
            gym.spaces.Box(low=-100, high=100, shape=(12,)),
            reward_fn,
            PLANT_INPUT_NAME,
            PLANT_OUTPUT_NAME,
        )