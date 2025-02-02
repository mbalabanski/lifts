import gymnasium.spaces as spaces
from gymnasium.envs.mujoco import mujoco_env

from typing import List
from ..filters import Filter


import mujoco

import numpy as np

# import core.physics


class LiftsEnv(mujoco_env.MujocoEnv):
    """
    Base environment for the Lifts project. This environment is a 3D environment where a quadcopter agent must safely place a payload on the target location.
    """

    metadata = {"render_modes": ['human', 'rgb_array', 'depth_array'], "render_fps": 100}
    bounds = np.array([[-5, -5, -5], [5, 5, 5]])

    def __init__(self, xml_path = "./lifts/assets/quadrotor.xml", render_mode = None, frame_skip=1, filters: List[Filter]=[]):
        
        self.target = self.generate_target_location()

        assert xml_path, "Must provide an XML Path."

        self.observation_space = spaces.Dict(
            {
                # agent space is its xyz coordinates and its xyz velocities
                "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(3,3), dtype=float),
                
                # payload space is the xyz coordinates of the payload and its xyz velocities
                "payload": spaces.Box(low=-np.inf, high=np.inf, shape=(3,3), dtype=float),

                # target space is the xyz coordinates of the target
                "target": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=float)
            }
        )

        super().__init__(xml_path, frame_skip=frame_skip, observation_space=self.observation_space)

        self.render_mode = render_mode

        # action space is motor actuator values
        self.action_space = spaces.Box(low=0, high=1.5, shape=(4,1), dtype=float)

        self.has_taken_off = False
        self.t = 0

        # split filters into action and state filters
        self.action_filters: List[Filter] = []
        self.state_filters: List[Filter] = []

        for filter in filters:
            if filter.action_filter:
                self.action_filters.append(filter)

            if filter.state_filter:
                self.state_filters.append(filter)

        self._prev_box_distance = None
        
    def get_sensor_data(self):
        return self.data.sensordata

    def get_physics_data(self):
        '''
        Fetches PV sensor data from Mujoco simulator.

        Should be in format:
            Position of quadrotor (3x1 vector)

            Linear Velocity of quadrotor (3x1 vector)

            Angular velocity of quadrotor (3x1 vector)

            Position of payload (3x1 vector)

            Linear velocity of quadrotor (3x1 vector)

            Angular velocity of quadrotor (3x1 vector)

        Returns: 6x3 vector of sensor data 
        '''
        return self.get_sensor_data()[:-1].reshape((-1, 3))
    
    def _get_contact_force(self):
        return self.get_sensor_data()[-1]

    def is_box_planted(self) -> bool:
        '''
        Returns whether or not the payload is planted on the ground.
        '''

        # box must have a contact force greater than tolerance for t time frames
        

        ACCEPTED_CONTACT_FORCE_TOLERANCE = 20

        return self._get_contact_force() > ACCEPTED_CONTACT_FORCE_TOLERANCE

        

    def _get_agent_position(self):
        return self.get_physics_data()[0]
    
    def _get_box_position(self):
        return self.get_physics_data()[3]

    def _get_obs(self):
        """
        Get the observation of the environment
        """

        pv_sensor_data = self.get_physics_data()

        return {
            "agent": pv_sensor_data[:3, :],
            "payload": pv_sensor_data[3:, :],
            "target": self.target
        }
    
    def _get_info(self):
        return {
            "has_taken_off": self.has_taken_off
        }
    
    def generate_target_location(self):
        """
        Generate a new target location.

        Base method places it at (2.0, 0, 0)

        Returns: location of the target
        """
        return np.array([2.0,0,0])
    
    def has_terminated(self):
        """
        Check if the environment has terminated.

        Should check for grid bounds, payload placement, and if drone has flipped over or stopped.
        """
        # check for planted box
        has_box_lifted = not self.is_box_planted()
        
        if not has_box_lifted and self.has_taken_off:
            return True # box has been planted

        if not self.has_taken_off and has_box_lifted:
            self.has_taken_off = True

        # check for exited bounds

        agent_pos = self._get_agent_position()
        

        if np.any(np.bitwise_or(agent_pos < self.bounds[0], agent_pos > self.bounds[1])):
            return True # exited bounds
        
        if self._get_agent_position()[2] < 0.05:
            return True

        return False

    
    def calculate_reward(self):
        """
        Calculate the reward for the current state.
        """
        # try different weights
        agent_pos = self._get_agent_position()

        # incentivize moving the box

        self._prev_box_distance

        return \
            + 1.5 * np.linalg.norm(self._get_box_position()) ** 2 \
            - (np.linalg.norm(self._get_box_position() - self.target) ** 2) \
            - 0.2 * (np.linalg.norm(self._get_obs()["agent"][2]) ** 2)  \
            - 5.0 * (agent_pos[2] - 2.0) ** 2 \

    
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each environment subclass.
        """

        self.set_state( # should be xyz, orientation (quaternion)
            np.array([0, 0, 0.6, 0, 0, 0, 0, 
                      0, 0, 0.1, 0, 0, 0, 0]),
            np.zeros((12,))
        )
        self.has_taken_off = False
        self.t = 0

        self._prev_box_distance = None

        return self._get_obs()

    def step(self, action):
        
        for filter in self.action_filters:
            action = filter.apply(action)
        
        if self.render_mode == 'human':
            self.render()

        self.t += 1

        self._step_mujoco_simulation(action, 1)

        reward = self.calculate_reward()
        observation = self._get_obs()
        terminated = self.has_terminated()
        truncated = False
        info = self._get_info()

        for filter in self.state_filters:
            observation = filter.apply(observation)

        return observation, reward, terminated, truncated, info
