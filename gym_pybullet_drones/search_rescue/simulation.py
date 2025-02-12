import os
import time
import argparse
import numpy as np
import pybullet as p
import cv2

from datetime import datetime
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from utils import drawVectorVelocity

class DroneSimulation:
    def __init__(self, **kwargs):
        self.drone_model = kwargs.get("drone", DroneModel("cf2x"))
        self.num_drones = kwargs.get("num_drones", 5)
        self.physics = kwargs.get("physics", Physics("pyb_gnd_drag_dw"))
        self.gui = kwargs.get("gui", True)
        self.record_video = kwargs.get("record_video", False)
        self.plot = kwargs.get("plot", False)
        self.user_debug_gui = kwargs.get("user_debug_gui", False)
        self.obstacles = kwargs.get("obstacles", False)
        self.simulation_freq_hz = kwargs.get("simulation_freq_hz", 240)
        self.control_freq_hz = kwargs.get("control_freq_hz", 48)
        self.duration_sec = kwargs.get("duration_sec", 9999999)
        self.output_folder = kwargs.get("output_folder", "results")
        self.colab = kwargs.get("colab", False)
        self.env = None
        self.logger = None
        self.controllers = []
        self.action = np.zeros((self.num_drones + 1, 4))  # Extra drone for manual control
        self.manual_drone_index = self.num_drones   # Index of the manual drone
        self.manual_drone_max_velocity = 0.05
        self.init_positions()
        self.init_trajectories()

    def init_positions(self):
        H = 0.5
        H_STEP = 0.05
        R = 0.3
        self.init_xyzs = np.array([
            [R * np.cos((i / 6) * 2 * np.pi + np.pi / 2), R * np.sin((i / 6) * 2 * np.pi + np.pi / 2) - R, H + i * H_STEP]
            for i in range(self.num_drones)
        ])
        self.init_rpys = np.array([[0, 0, i * (np.pi / 2) / self.num_drones] for i in range(self.num_drones)])
        self.drone_manual_position = np.array([1, 1, 0.5]) # Initial position of the manual drone
        self.init_xyzs = np.vstack((self.init_xyzs, self.drone_manual_position))

    def init_trajectories(self):
        PERIOD = 10
        NUM_WP = self.control_freq_hz * PERIOD
        TARGET_POS = np.zeros((NUM_WP, 3))
        R = 0.5
        for i in range(NUM_WP):
            TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+self.init_xyzs[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+self.init_xyzs[0, 1], self.init_xyzs[0, 2] 


        self.waypoints = TARGET_POS
        self.wp_counters = np.array([int((i * NUM_WP / 6) % NUM_WP) for i in range(self.num_drones)])

    def setup_environment(self):
        self.env = CtrlAviary(
            drone_model=self.drone_model,
            num_drones=self.num_drones + 1,
            initial_xyzs=self.init_xyzs,
            initial_rpys=np.vstack((self.init_rpys, [0, 0, 0])),
            physics=self.physics,
            neighbourhood_radius=10,
            pyb_freq=self.simulation_freq_hz,
            ctrl_freq=self.control_freq_hz,
            gui=self.gui,
            record=self.record_video,
            obstacles=self.obstacles,
            user_debug_gui=self.user_debug_gui,
        )
        self.logger = Logger(
            logging_freq_hz=self.control_freq_hz,
            num_drones=self.num_drones + 1,
            output_folder=self.output_folder,
            colab=self.colab,
        )
        self.controllers = [DSLPIDControl(drone_model=self.drone_model) for _ in range(self.num_drones)]
        p.addUserDebugParameter("Manual Drone Velocity",0,1,0.5)

        # Controller for the manual drone
        self.manual_controller = DSLPIDControl(drone_model=self.drone_model) 

    def run_simulation(self):
        """
        Runs the drone simulation for a specified duration, updating the environment and drone controls at each step.
        """

        self.setup_environment()  # Initialize the simulation environment

        start_time = time.time()  # Record the simulation start time

        # Calculate the total number of simulation steps based on duration and control frequency
        num_steps = int(self.duration_sec * self.env.CTRL_FREQ)

        for step_num in range(num_steps):  # Iterate through each simulation step
            # Execute a step in the environment, applying the current action
            obs, _, _, _, _ = self.env.step(self.action)

            # Update drone controls based on the current observation from the environment
            self.update_controls(obs)

            # Visualize the velocity vector for the manual drone
            drawVectorVelocity(self.num_drones, obs[self.num_drones], p) 

            # Process any user input or other event handling
            self.process_input()

            ## Camera view ========================================
            #self.use_camera()

            # Synchronize the simulation with real-time (maintain desired control frequency)
            sync(step_num, start_time, self.env.CTRL_TIMESTEP)

        # Clean up the environment after the simulation is complete
        self.env.close()

        # Generate and display plots if requested
        if self.plot:
            self.logger.plot()  # Generate plots using the logger object

    def use_camera(self):
        ## Camera view ========================================
        # Camera settings
        width, height = 320, 240
        fov = 60
        aspect = width / height
        near, far = 0.1, 10

        camera_position = [1, 1, 1]
        target_position = [0, 0, 0]
        up_vector = [0, 0, 1]

        view_matrix = p.computeViewMatrix(cameraEyePosition=camera_position,
                                            cameraTargetPosition=target_position,
                                            cameraUpVector=up_vector)

        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, projection_matrix)
            
        # Convert image to correct format
        rgb_array = np.array(rgb, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]

        # Show the image in a separate OpenCV window
        cv2.imshow("PyBullet Camera View", cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))


    def update_controls(self, obs):
        for j in range(self.num_drones):
            target_index = self.wp_counters[j]  # Get the current waypoint index
            target_pos = self.waypoints[target_index]  # Get the target position from the trajectory

            self.action[j, :], _, _ = self.controllers[j].computeControlFromState(
                control_timestep=self.env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=target_pos,
                target_rpy=self.init_rpys[j, :],
            )

            self.wp_counters[j] = (self.wp_counters[j] + 1) % len(self.waypoints)  # Move to the next waypoint

        # Update the control for the manual drone
        self.action[self.manual_drone_index, :] = self.manual_controller.computeControlFromState(
            control_timestep=self.env.CTRL_TIMESTEP,
            state=obs[self.manual_drone_index],
            target_pos=self.drone_manual_position,
            target_rpy=[0, 0, 0],
        )[0]  # Extract the control action from the returned tuple


    def process_input(self):
        # get the value of the user debug parameter
        self.manual_drone_max_velocity = p.readUserDebugParameter(0)
        
        step_size = self.manual_drone_max_velocity * 0.1

        key_events = p.getKeyboardEvents()
        if p.B3G_LEFT_ARROW in key_events:
            self.drone_manual_position[0] -= step_size
        if p.B3G_RIGHT_ARROW in key_events:
            self.drone_manual_position[0] += step_size
        if p.B3G_UP_ARROW in key_events:
            self.drone_manual_position[1] += step_size
        if p.B3G_DOWN_ARROW in key_events:
            self.drone_manual_position[1] -= step_size
        if ord('u') in key_events:
            self.drone_manual_position[2] += step_size
        if ord('d') in key_events:
            self.drone_manual_position[2] -= step_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control Drone using PID and arrow keys")
    parser.add_argument("--drone", default=DroneModel("cf2x"), type=DroneModel, choices=DroneModel)
    parser.add_argument("--num_drones", default=5, type=int)
    parser.add_argument("--physics", default=Physics("pyb_gnd_drag_dw"), type=Physics, choices=Physics)
    parser.add_argument("--gui", default=True, type=str2bool)
    parser.add_argument("--record_video", default=False, type=str2bool)
    parser.add_argument("--plot", default=False, type=str2bool)
    parser.add_argument("--user_debug_gui", default=False, type=str2bool)
    parser.add_argument("--obstacles", default=True, type=str2bool)
    parser.add_argument("--simulation_freq_hz", default=240, type=int)
    parser.add_argument("--control_freq_hz", default=48, type=int)
    parser.add_argument("--duration_sec", default=9999999, type=int)
    parser.add_argument("--output_folder", default="results", type=str)
    parser.add_argument("--colab", default=False, type=bool)
    args = parser.parse_args()
    
    sim = DroneSimulation(**vars(args))
    sim.run_simulation()
