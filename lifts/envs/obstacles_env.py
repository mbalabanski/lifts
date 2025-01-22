import lifts.envs.lifts_env as lifts_env

import gymnasium.spaces as spaces
import numpy as np



class LiftsObstacleEnv(lifts_env.LiftsEnv):
    metadata = {"render_modes": ['human', 'rgb_array', 'depth_array'], "render_fps": 100}
    bounds = np.array([[-5, -5, -5], [5, 5, 5]])

    def __init__(self, xml_path, render_mode, frame_skip=1):
        super().__init__(xml_path, render_mode, frame_skip=frame_skip)

        self.observation_space = spaces.Dict(
            {
                # agent space is its xyz coordinates and its xyz velocities
                "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(6,1), dtype=float),
                
                # payload space is the xyz coordinates of the payload and its xyz velocities
                "payload": spaces.Box(low=-np.inf, high=np.inf, shape=(6,1), dtype=float),

                # target space is the xyz coordinates of the target
                "target": spaces.Box(low=-np.inf, high=np.inf, shape=(3,1), dtype=float),

                
            }
        )

