"""
Multi-agent to single agent wrapper loading a policy to other agents
"""
from gym import spaces

from gym_duckietown.multiagent_simulator import MultiagentDuckiebotEnv


class MultiToSingle(MultiagentDuckiebotEnv):
    def __init__(self, multi_env, collaborator_policy=None):
        self.multi_env = multi_env
        self.collaborator_policy = collaborator_policy
        self.observation_space = spaces.Box()
        self.action_space = spaces.Box()

    def step(self, action):
        """step function"""
        raise NotImplementedError

    def reset(self):
        """reset function"""
        raise NotImplementedError

    def _calculate_reward(self):
        """calculate reward for all agent sin the environment"""
        raise NotImplementedError
