#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : _wrapper.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

try:
    import gym
except ImportError:
    from jacinle.logging import get_logger
    logger = get_logger(__file__)
    logger.warning('Cannot import gym.')

    gym = None

# https://github.com/ppaquette/gym-super-mario/blob/master/ppaquette_gym_super_mario/wrappers/action_space.py
from ..gym_adapter import DiscreteToMultiDiscrete


class GymNintendoWrapper(gym.Wrapper):
    """
        Wrapper to convert MultiDiscrete action space to Discrete

        Only supports one config, which maps to the most logical discrete space possible
    """
    def __init__(self, env):
        super().__init__(env)
        # Nintendo Game Controller
        mapping = {
            0: [0, 0, 0, 0, 0, 0],  # NOOP
            1: [1, 0, 0, 0, 0, 0],  # Up
            2: [0, 0, 1, 0, 0, 0],  # Down
            3: [0, 1, 0, 0, 0, 0],  # Left
            4: [0, 1, 0, 0, 1, 0],  # Left + A
            5: [0, 1, 0, 0, 0, 1],  # Left + B
            6: [0, 1, 0, 0, 1, 1],  # Left + A + B
            7: [0, 0, 0, 1, 0, 0],  # Right
            8: [0, 0, 0, 1, 1, 0],  # Right + A
            9: [0, 0, 0, 1, 0, 1],  # Right + B
            10: [0, 0, 0, 1, 1, 1],  # Right + A + B
            11: [0, 0, 0, 0, 1, 0],  # A
            12: [0, 0, 0, 0, 0, 1],  # B
            13: [0, 0, 0, 0, 1, 1],  # A + B
        }
        self.action_space = DiscreteToMultiDiscrete(self.action_space, mapping)

    def _step(self, action):
        return self.env._step(self.action_space(action))
