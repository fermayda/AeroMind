import gymnasium as gym

from pydrake.all import (
    DiagramBuilder, SceneGraph, Simulator, System, Context,
    MeshcatVisualizer, EventStatus
)
from pydrake.examples import QuadrotorPlant, QuadrotorGeometry
from pydrake.gym._drake_gym_env import DrakeGymEnv
import numpy as np

class QuadrotorEnv:
    @staticmethod
    def build(meshcat, target_position: np.ndarray, target_velocity: np.ndarray, realtime_rate=-1):
        # State is x ,y , z, roll, pitch, yaw + velocities.
        builder = DiagramBuilder()

        plant = builder.AddSystem(QuadrotorPlant())


        PLANT_INPUT_NAME = "plant_input"
        PLANT_OUTPUT_NAME = "state"
        builder.ExportInput(plant.get_input_port(0), PLANT_INPUT_NAME)
        builder.ExportOutput(plant.get_output_port(0), PLANT_OUTPUT_NAME)


        # Set up visualization in MeshCat
        scene_graph = builder.AddSystem(SceneGraph())
        QuadrotorGeometry.AddToBuilder(builder, plant.get_output_port(0), scene_graph)
        if meshcat is not None:
            meshcat.Delete()
            meshcat.ResetRenderMode()
            meshcat.SetProperty("/Background", "visible", False)
            MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
            input("Press enter once meshcat has finished initializing and you're ready")
        # end setup for visualization

        diagram = builder.Build()

        # Set up a simulator to run this diagram
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(realtime_rate)


        def reward_fn(system: System, context: Context):
            state = system.get_output_port(0).Eval(context) 
            # print(state.shape, state[:3])
            pos_err = np.linalg.norm(state[:3] - target_position)


            yaw_rate = state[11]

            
            # combine position, yaw rate, and roll/pitch error
            return 1-(pos_err + 0.01*yaw_rate**2 + np.linalg.norm(state[6:8]))

        def monitor_fn(context: Context) -> EventStatus:
            system = simulator.get_system()
            state = system.get_output_port(0).Eval(context)
            if np.linalg.norm(state[:3]) > 3:
                return EventStatus.ReachedTermination(system, 'too far from origin')
            if state[3]<-1:
                return EventStatus.ReachedTermination(system, 'height too low')
            return EventStatus.DidNothing()

        simulator.set_monitor(monitor_fn)

        return DrakeGymEnv(
            simulator, 
            1/100, 
            gym.spaces.Box(low=0, high=10, shape=(4,)),
            gym.spaces.Box(low=-300, high=300, shape=(12,)),
            reward_fn,
            PLANT_INPUT_NAME,
            PLANT_OUTPUT_NAME,
        )