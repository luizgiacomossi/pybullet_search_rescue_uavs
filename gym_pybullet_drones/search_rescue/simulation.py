import os
import time
import argparse
import numpy as np
import pybullet as p
import cv2
import matplotlib.pyplot as plt

from datetime import datetime
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from utils import drawVectorVelocity

import numpy as np
import pybullet as p
from typing import List

class Vehicle:
    def __init__(self, model, index: int, initial_position: List[float], initial_orientation: List[float]):
        """
        Initialize the vehicle with a model, index, position, and orientation.
        
        :param model: The model of the vehicle.
        :param index: The unique index of the vehicle.
        :param initial_position: Initial position [x, y, z] of the vehicle.
        :param initial_orientation: Initial orientation [roll, pitch, yaw] of the vehicle.
        """
        self.model = model
        self.index = index
        self.position = np.array(initial_position)
        self.orientation = np.array(initial_orientation)
        self.target_position = np.array(initial_position)
        self.target_orientation = np.array(initial_orientation)
        self.model_urdf = None
        self.action = np.zeros(4)

    def set_position(self, position: List[float]):
        """Update the vehicle's position."""
        self.target_position = np.array(position)

    def set_orientation(self, orientation: List[float]):
        """Update the vehicle's orientation."""
        self.target_orientation = np.array(orientation)
    
    def set_pose(self, position_xyz, orientation_rpy):
        self.target_position = position_xyz
        self.target_orientation = orientation_rpy

    def get_position(self) -> np.ndarray:
        """Return the current position of the vehicle."""
        return self.position
    
    def get_orientation(self) -> np.ndarray:
        """Return the current orientation of the vehicle."""
        return self.orientation

class Drone(Vehicle):
    MAX_VELOCITY = 5  # m/s
    MAX_ASCENT_RATE = 2  # m/s
    MAX_DESCENT_RATE = 1  # m/s, to prevent instability
    BATTERY_LIFE_HOVERS = 7  # minutes
    BATTERY_LIFE_AGRESSIVE = 5  # minutes

    def __init__(self, model: 'DroneModel' = None, index: int = 0, initial_position: List[float] = [0, 0, 0], 
                 initial_orientation: List[float] = [0, 0, 0]):
        """
        Initialize a drone with the given model, index, position, and orientation.
        
        :param model: The drone model. Defaults to 'cf2x' if not provided.
        :param index: The unique index of the drone.
        :param initial_position: Initial position [x, y, z] of the drone.
        :param initial_orientation: Initial orientation [roll, pitch, yaw] of the drone.
        """
        model = model if model is not None else DroneModel("cf2x")
        super().__init__(model, index, initial_position, initial_orientation)

        # Load the URDF model for the drone
        #self.model_urdf = p.loadURDF('cf2x.urdf', self.position.tolist(), 
        #                             p.getQuaternionFromEuler(self.orientation.tolist()), 
        #                             flags=p.URDF_USE_INERTIA_FROM_FILE)

        self.battery_level = 100
        self.controller = DSLPIDControl(drone_model=model)

        print(f"Drone {index} initialized")

    def compute_control(self, control_timestep: float, state: np.ndarray, target_pos: np.ndarray, target_rpy: np.ndarray) -> np.ndarray:
        """
        Compute control signals for the drone.
        
        :param control_timestep: The timestep to apply control.
        :param state: The current state of the drone.
        :param target_pos: The target position for the drone.
        :param target_rpy: The target roll, pitch, yaw for the drone.
        :return: The computed control signals.
        """
        return self.controller.computeControlFromState(control_timestep, state, self.target_position, self.target_orientation)

    def update_control(self, action: np.ndarray):
        """Update the control action for the drone."""
        self.action = action

class Swarm:
    def __init__(self):
        """Initialize the drone swarm."""
        self.drones = []
        self.env = None
        self.num_drones = 0

    def add_drone(self, drone):
        self.drones.append(drone)
    
    def set_env(self, env):
        self.env = env
        self.num_drones = env.NUM_DRONES

    def create_drone(self, n_drones = 1):
        for i in range(n_drones):
            pass

    def change_color_drones(self):
        # set a random color for the drones
        for i in range(self.num_drones ):
            print(f"Changing color of drone {i}")
            p.changeVisualShape(self.env.DRONE_IDS[i], -1, rgbaColor=[np.random.rand(), np.random.rand(), np.random.rand(), 1])

    def go_to_position(self,id, position_xyz):
        pass

class DroneSimulation:
    def __init__(self, **kwargs):
        self.drone_model = kwargs.get("drone", DroneModel("cf2x"))
        self.num_drones = kwargs.get("num_drones", 5)
        self.physics = kwargs.get("physics", Physics("pyb_gnd_drag_dw"))
        self.gui = kwargs.get("gui", True)
        self.record_video = kwargs.get("record_video", False)
        self.plot = kwargs.get("plot", False)
        self.user_debug_gui = kwargs.get("user_debug_gui", True)
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
        self.manual_drone_rpy = np.array([0,0,0.1]) 

        self.swarm = Swarm()

        self.camera_follow_drone = self.num_drones 
        self.init_positions()
        self.init_trajectories()

    def create_drones(self):

        # creates swarm
        self.env = CtrlAviary(
            drone_model=self.drone_model,   # Model of the drone
            num_drones=self.num_drones + 1, # Number of drones (including the manual drone)
            initial_xyzs=self.init_xyzs,   # Initial positions of the drones
            initial_rpys=np.vstack((self.init_rpys, [0, 0, 0])), # Initial orientations of the drones
            physics=self.physics, # Physics parameters
            neighbourhood_radius=10, # Neighbourhood radius for the drones
            pyb_freq=self.simulation_freq_hz, # PyBullet simulation frequency
            ctrl_freq=self.control_freq_hz, # Control frequency
            gui=self.gui, # Enable the GUI 
            record=self.record_video, # Record the video 
            obstacles=self.obstacles, # Include obstacles in the environment 
            user_debug_gui=self.user_debug_gui, # Enable the user debug GUI
            vision_attributes = True # Enable vision attributes
        )

        self.swarm.set_env(self.env)
        self.swarm.change_color_drones()

        print("Drones created")
        print(f"Number of drones: {self.num_drones}")
        print(f"Drones: {self.env.getDroneIds()}")

    def init_positions(self):
        H = 0.5
        H_STEP = 0.05
        R = 0.3
        self.init_xyzs = np.array([
            [R * np.cos((i / 6) * 2 * np.pi + np.pi / 2), R * np.sin((i / 6) * 2 * np.pi + np.pi / 2) - R, H + i * H_STEP]
            for i in range(self.num_drones)
        ])
        self.init_rpys = np.array([[0, 0, i * (np.pi / 2) / self.num_drones] for i in range(self.num_drones)])

        # Initial position of the manual drone
        self.drone_manual_position = np.array([1, 1, 0.5]) 
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
        ## Initialize the simulation environment

        # create drones
        self.create_drones()
 
        # Initialize the logger object to record data during the simulation 
        self.logger = Logger(
            logging_freq_hz=self.control_freq_hz,
            num_drones=self.num_drones + 1,
            output_folder=self.output_folder,
            colab=self.colab,
        )

        # Controller for the manual drone
        self.drone = Drone( 
                            model=self.drone_model, 
                            index=self.num_drones+1, 
                            initial_position=[0, 0, 0.5], 
                            initial_orientation=[0, 0, 0]
                          )
        
        self.manual_controller = DSLPIDControl(drone_model=self.drone_model) 

        # Initialize the controllers for the drones

        self.controllers = [DSLPIDControl(drone_model=self.drone_model) for _ in range(self.num_drones)]
        p.addUserDebugParameter("Manual Drone Velocity",0,1,0.5)

        # Set up the camera (adjust parameters as needed)
        self.cameraDistance = 3
        self.cameraYaw = 90
        self.cameraPitch = -30
        self.cameraTargetPosition = [0, 0, 0]
        p.resetDebugVisualizerCamera(self.cameraDistance, self.cameraYaw, self.cameraPitch, self.cameraTargetPosition)

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

            # camera follows the manual drone
            # gets the position of the manual drone
            p.resetDebugVisualizerCamera(1, 0, -30, obs[self.camera_follow_drone][0:3])

        # Clean up the environment after the simulation is complete
        #self.env.close()

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
        for j in range(self.num_drones ):
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
                                                    control_timestep=self.env.CTRL_TIMESTEP, # Control timestep
                                                    state=obs[self.manual_drone_index], # Current state of the manual drone
                                                    target_pos=self.drone_manual_position, # Target position for the manual drone
                                                    target_rpy= self.manual_drone_rpy, # Target roll, pitch, yaw
                                                )[0]  # Extract the control action from the returned tuple
        #self.manual_drone_rpy[2]= self.manual_drone_rpy[2] + 0.1

        #print(f"Manual Drone desired Position: {self.drone_manual_position}, {obs[self.manual_drone_index][0:3]}")
        #print(f"Error: {np.linalg.norm(self.drone_manual_position - obs[self.manual_drone_index][0:3])}")
        #print(f"Manual Drone Position: {obs[self.manual_drone_index][0:3]}")

    def process_input(self):
        # get the value of the user debug parameter
        try:
            self.manual_drone_max_velocity = p.readUserDebugParameter(0)
        except:
            pass
        
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
        if ord('r') in key_events:
            # reset the manual drone position
            self.drone_manual_position = np.array([1, 1, 0.5])
        
        # if n is pressed and the camera is following a drone
        # only changes when released
        if ord('n') in key_events and key_events[ord('n')] == 4:
            # will change camera view to next drone
            self.camera_follow_drone += 1
            print(f"Camera following drone {self.camera_follow_drone}")
            if self.camera_follow_drone >= self.num_drones + 1:
                self.camera_follow_drone = 0


        # Handle mouse input
        mouse_events = p.getMouseEvents()
        getDebugVisualizerCamera = p.getDebugVisualizerCamera()
        camera_position_x,camera_position_y = getDebugVisualizerCamera[0:2]


        if len(mouse_events) > 0:
            for event in mouse_events:
                mouse_x = event[1]
                mouse_y = event[2]
                if event[3] == 0 and event[4] == 3:  # left mouse button pressed 
                    print(f'Left mouse button pressed at ({mouse_x}, {mouse_y})')
                    # use the mouse position to set the target position of the manual drone
                    #self.drone_manual_position[0] = (mouse_x ) / camera_position_x
                    #self.drone_manual_position[1] = (mouse_y) / camera_position_y

                if event[3] == 2 and event[4] == 3: # Right mouse button pressed

                    print(f'Right mouse button pressed at ({mouse_x}, {mouse_y})')
                    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
                    #p.configureDebugVisualizer(shadowMapResolution= 32768)

    def camera_drone(self, drone_id):
        '''Camera view of the drone'''

        rgb, dep, seg = self.env._getDroneImages(drone_id)
        # show the image in a separate OpenCV window
        plt.imshow(rgb)  # Displays the RGB image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control Drone using PID and arrow keys")
    parser.add_argument("--drone", default=DroneModel("cf2x"), type=DroneModel, choices=DroneModel)
    parser.add_argument("--num_drones", default=3, type=int)
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
