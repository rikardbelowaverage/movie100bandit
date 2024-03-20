""" Agents take actions in an environment.

All agents should inherit from the Agent class and work from there.
"""

from __future__ import division
from __future__ import print_function
import logging

class Agent(object):
    """Base class for all bandit agents."""

    def __init__(self):
        """Initialize the agent."""
        self.logger = logging.getLogger(name='AgentLogger')
        self.logger.setLevel(logging.WARNING)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def update_observation(self, observation, action, reward):
        """Add an observation to the records."""
        pass

    def pick_action(self, observation):
        """Select an action based upon the policy + observation."""
        pass