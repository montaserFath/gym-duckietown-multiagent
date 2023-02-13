"""
Duckietown wit Multi-agent (multi-duckiebot)
"""
import geometry
import gym
import numpy as np
from gym import spaces

from gym_duckietown.simulator import Simulator
from .exceptions import InvalidMapException, NotInLane
from . import logger
from simulator import _update_pos
from duckietown_world import MapFormat1Constants


class MultiagentSimulator(Simulator):
    """
    Multiagent class
    """

    def __init__(self, n_agents=2, **kwargs):
        self._n_agents = n_agents
        Simulator.__init__(self, **kwargs)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self._n_agents, self.camera_height, self.camera_width, 3), dtype=np.uint8
        )
        self.last_action, self.wheelVels = {}, {}
        self.cur_pos, self.cur_angle = {}, {}
        self.speed, self.timestamp = {}, {}
        self.state = {}
        for agent_id in range(self._n_agents):
            self.last_action.update({agent_id: np.array([0, 0])})
            self.wheelVels.update({agent_id: np.array([0, 0])})
            self.cur_pos.update({agent_id: np.array([0, 0, 0])})
            self.cur_angle.update({agent_id: 0.0})
            self.speed.update({agent_id: 0.0})
            self.timestamp.update({agent_id: 0.0})

    def reset(self, segment: bool = False):
        """Reset function"""
        self.step_count = 0
        for agent_id in range(self._n_agents):
            self.speed.update({agent_id: 0.0})
            self.timestamp.update({agent_id: 0.0})

        self.cur_pos, self.cur_angle, p = self._prepare_env_reset(n_agent=self._n_agents)
        for agent_id in range(self._n_agents):
            q = self.cartesian_from_weird(self.cur_pos[agent_id], self.cur_angle[agent_id])
            v0 = geometry.se2_from_linear_angular(np.array([0, 0]), 0)
            c0 = q, v0
            self.state = p.initialize(c0=c0, t0=0)

        logger.info(f"Starting at {self.cur_pos} {self.cur_angle}")

        # Generate the first camera image
        obs = self.render_obs(segment=segment)

        # Return first observation
        return obs

    def update_physics(self, action: dict, delta_time: float = None):
        """update physics for all agents"""
        if delta_time is None:
            delta_time = self.delta_time
        prev_pos = self.cur_pos.copy()
        for agent_id in range(self._n_agents):
            self.wheelVels[agent_id] = action[agent_id] * self.robot_speed * 1
            self.timestamp += delta_time

            # Update the robot's position
            self.cur_pos[agent_id], self.cur_angle[agent_id] = _update_pos(self, action[agent_id])
            self.last_action[agent_id] = action[agent_id]
            delta_pos = self.cur_pos[agent_id] - prev_pos
            self.speed[agent_id] = np.linalg.norm(delta_pos) / delta_time
        self.step_count += 1

        # Update world objects
        for obj in self.objects:
            if obj.kind == MapFormat1Constants.KIND_DUCKIEBOT:
                if not obj.static:
                    obj_i, obj_j = self.get_grid_coords(obj.pos)
                    same_tile_obj = [
                        o
                        for o in self.objects
                        if tuple(self.get_grid_coords(o.pos)) == (obj_i, obj_j) and o != obj
                    ]

                    obj.step_duckiebot(delta_time, self.closest_curve_point, same_tile_obj)
            else:
                # print("stepping all objects")
                obj.step(delta_time)

    def get_agent_info(self) -> dict:
        """"""
        info, misc = {}, {}
        for agent_id in range(self._n_agents):
            info.update({agent_id: {}})
        for agent_id in range(self._n_agents):
            info[agent_id] = {"action": self.last_action[agent_id]}
            pos, angle = self.cur_pos[agent_id], self.cur_angle[agent_id]
            if self.full_transparency:
                try:
                    lp = self.get_lane_pos2(pos, angle)
                    info[agent_id].update({"lane_position": lp.as_json_dict()})
                except NotInLane:
                    pass
            info[agent_id].update({'robot_speed': self.speed[agent_id]})
            info[agent_id].update({"proximity_penalty": self.proximity_penalty2(pos, angle)})
            info[agent_id].update({"cur_pos": [float(pos[0]), float(pos[1]), float(pos[2])]})
            info[agent_id].update({"wheel_velocities": [self.wheelVels[agent_id][0], self.wheelVels[agent_id][1]]})
            info[agent_id].update({"cur_angle": float(angle)})
            info[agent_id].update({"timestamp": self.timestamp[agent_id]})
            info[agent_id].update({"tile_coords": list(self.get_grid_coords(pos))})
        misc["Simulator"] = info
        return misc

    def step(self, action: dict):
        """step function for multi-agent env"""
        for agent_id in range(self._n_agents):
            action[agent_id] = np.clip(action[agent_id], -1, 1)

        for _ in range(self.frame_skip):
            self.update_physics(action)

        # Generate the current camera image
        obs = self.render_obs()
        misc = self.get_agent_info()

        d = self._compute_done_reward()
        misc["Simulator"]["msg"] = d.done_why

        return obs, d.reward, d.done, misc
