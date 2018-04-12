from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import gym
import gym_fish

from collections import deque

import modules.agent as agent
import modules.observer as observer
from modules.parameters import *
from modules.main import *


if __name__ == "__main__":
    key = 'fish-v0'
    exp = Experiment(key)
    agent = agent.DQNAgent(exp.env)
    epsilon = observer.EpsilonUpdater(agent)
    agent.add_observer(epsilon)
    exp.run_experiment(agent)
    