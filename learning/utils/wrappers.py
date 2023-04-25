import gym
from gym import spaces
import numpy as np
import cv2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn

from gym_duckietown.simulator import Simulator


class MotionBlurWrapper(Simulator):
    def __init__(self, env=None):
        Simulator.__init__(self)
        self.env = env
        self.frame_skip = 3
        self.env.delta_time = self.env.delta_time / self.frame_skip

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        # Actions could be a Python list
        action = np.array(action)
        motion_blur_window = []
        for _ in range(self.frame_skip):
            obs = self.env.render_obs()
            motion_blur_window.append(obs)
            self.env.update_physics(action)

        # Generate the current camera image

        obs = self.env.render_obs()
        motion_blur_window.append(obs)
        obs = np.average(motion_blur_window, axis=0, weights=[0.8, 0.15, 0.04, 0.01])

        misc = self.env.get_agent_info()

        d = self.env._compute_done_reward()
        misc["Simulator"]["msg"] = d.done_why

        return obs, d.reward, d.done, misc


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3), crop_top=False):
        super(ResizeWrapper, self).__init__(env)
        # self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            # (360, 640, 3),
            dtype=self.observation_space.dtype,
        )
        self.shape = shape
        self.crop_top = crop_top

    def observation(self, observation):
        # return observation[120:] if self.crop_top else observation
        return cv2.resize(observation[120:] if self.crop_top else observation, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            # [obs_shape[0], obs_shape[1], obs_shape[2]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        # return observation
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -1
        # elif reward > 0:
        #     reward /= 10
        else:
            reward /= 10.0

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_


# this a reward wrapper to crop the reward to be between -1 and +1
class RewardCropWrapper(gym.RewardWrapper):
    def __int__(self, env):
        super(RewardCropWrapper, self).__int__(env)
        self.reward_range(-1, 1)

    def reward(self, reward):
        return reward / 1000


class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env, stop_action=False):
        gym.ActionWrapper.__init__(self, env)
        self.stop_action = stop_action
        self.action_space = spaces.Discrete(4 if self.stop_action else 3)

    def action(self, action):
        #  Turn left
        if action == 0:
            vels = [0.6, 1.0]
        # Turn right
        elif action == 1:
            vels = [0.6, -1.0]
        # Go forward
        elif action == 2:
            vels = [0.5, 0.0]
        # stop
        elif action == 3:
            vels = [0.0, 0.0]
        else:
            assert False, "unknown action"

        return np.array(vels)

    def reverse_action(self, action):
        raise NotImplementedError()


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
