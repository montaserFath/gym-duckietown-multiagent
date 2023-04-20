# coding=utf-8
from abc import ABC

import cv2
import gym
import numpy as np
from gym import spaces


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
        wheel_distance = 0.102
        min_rad = 0.08

        # Turn left
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


class SteeringToWheelVelWrapper(gym.ActionWrapper):
    """
    Converts policy that was trained with [velocity|heading] actions to
    [wheelvel_left|wheelvel_right] to comply with AIDO evaluation format
    """

    def __init__(self, env, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0):
        gym.ActionWrapper.__init__(self, env)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

    def action(self, action):
        vel, angle = action

        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels

    def reverse_action(self, action):
        raise NotImplementedError()


class PyTorchObsWrapper(gym.ObservationWrapper):
    """
    Transpose the observation image tensors for PyTorch
    """

    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 1, 0)


# class ResizeWrapper(gym.ObservationWrapper):
#     def __init__(self, env=None, resize_w=80, resize_h=80):
#         gym.ObservationWrapper.__init__(self, env)
#         self.resize_h = resize_h
#         self.resize_w = resize_w
#         obs_shape = self.observation_space.shape
#         self.observation_space = spaces.Box(
#             self.observation_space.low[0, 0, 0],
#             self.observation_space.high[1, 1, 1],
#             [obs_shape[0], resize_h, resize_w],
#             dtype=self.observation_space.dtype,
#         )
#
#     def observation(self, observation):
#         return observation
#
#     def reset(self):
#         obs = gym.ObservationWrapper.reset(self)
#         return cv2.resize(
#             obs.swapaxes(0, 2), dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC
#         ).swapaxes(0, 2)
#
#     def step(self, actions):
#         obs, reward, done, info = gym.ObservationWrapper.step(self, actions)
#         return (
#             cv2.resize(
#                 obs.swapaxes(0, 2), dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC
#             ).swapaxes(0, 2),
#             reward,
#             done,
#             info,
#         )

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
            # [obs_shape[2], obs_shape[0], obs_shape[1]],
            [obs_shape[0], obs_shape[1], obs_shape[2]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation
        # return observation.transpose(2, 0, 1)


class RewardCropWrapper(gym.RewardWrapper):
    def __int__(self, env):
        super(RewardCropWrapper, self).__int__(env)
        self.reward_range(-1, 1)

    def reward(self, reward):
        return reward / 1000


class UndistortWrapper(gym.ObservationWrapper):
    """
    To Undo the Fish eye transformation - undistorts the image with plumbbob distortion
    Using the default configuration parameters on the duckietown/Software repo
    https://github.com/duckietown/Software/blob/master18/catkin_ws/src/
    ...05-teleop/pi_camera/include/pi_camera/camera_info.py
    """

    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)

        assert env.unwrapped.distortion, "Distortion is false, no need for this wrapper"

        # Set a variable in the unwrapped env so images don't get distorted
        self.env.unwrapped.undistort = True

        # K - Intrinsic camera matrix for the raw (distorted) images.
        camera_matrix = [
            305.5718893575089,
            0,
            303.0797142544728,
            0,
            308.8338858195428,
            231.8845403702499,
            0,
            0,
            1,
        ]
        self.camera_matrix = np.reshape(camera_matrix, (3, 3))

        # distortion parameters - (k1, k2, t1, t2, k3)
        distortion_coefs = [-0.2, 0.0305, 0.0005859930422629722, -0.0006697840226199427, 0]
        self.distortion_coefs = np.reshape(distortion_coefs, (1, 5))

        # R - Rectification matrix - stereo cameras only, so identity
        self.rectification_matrix = np.eye(3)

        # P - Projection Matrix - specifies the intrinsic (camera) matrix
        #  of the processed (rectified) image
        projection_matrix = [
            220.2460277141687,
            0,
            301.8668918355899,
            0,
            0,
            238.6758484095299,
            227.0880056118307,
            0,
            0,
            0,
            1,
            0,
        ]
        self.projection_matrix = np.reshape(projection_matrix, (3, 4))

        # Initialize mappings

        # Used for rectification
        self.mapx = None
        self.mapy = None

    def observation(self, observation):
        return self._undistort(observation)

    def _undistort(self, observation):
        if self.mapx is None:
            # Not initialized - initialize all the transformations we'll need
            self.mapx = np.zeros(observation.shape)
            self.mapy = np.zeros(observation.shape)

            H, W, _ = observation.shape

            # Initialize self.mapx and self.mapy (updated)
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion_coefs,
                self.rectification_matrix,
                self.projection_matrix,
                (W, H),
                cv2.CV_32FC1,
            )

        return cv2.remap(observation, self.mapx, self.mapy, cv2.INTER_NEAREST)


class RLSimpleObservation(gym.ObservationWrapper):
    """
    Given an image from the camera, it detects lines, and convert it to the camera coordinates then robot's coordinates
    Observations:
    0: (float) distance to the left line edge
    1: (float) distance to the right line edge
    2: (float) distance to central line
    3-5: (list) bot current position in (x, y, z)
    6: (float) bot current linear velocity
    7: (binary) is the bot in the right or left line (in two ways)
    8: (binary) is the bot inside the road course or not

    """
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(9,),  dtype=np.float32)

    def get_observation(self):
        """get observation"""
        raise NotImplementedError
