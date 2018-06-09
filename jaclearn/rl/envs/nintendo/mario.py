#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mario.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import copy

from ._wrapper import GymNintendoWrapper
from ..gym import GymRLEnv

try:
    import gym
    import ppaquette_gym_super_mario
    from ppaquette_gym_super_mario import wrappers
except ImportError:
    from jacinle.logging import get_logger
    logger = get_logger(__file__)
    logger.warning('Cannot import gym and ppaquette_gym_super_mario.')

    gym = None
    ppaquette_gym_super_mario = None


class GymMarioRLEnv(GymRLEnv):
    """
        Using https://github.com/ppaquette/gym-super-mario/tree/gabegrand

        dhh: use meta-env and change_level to hack restart,
            old restart might restore to a non-start intermediate state
    """
    def __init__(self, name, dump_dir=None, force_dump=False, state_mode='DEFAULT'):
        super().__init__(name, dump_dir, force_dump, state_mode)

        self._cur_iter = -1

    def _make_env(self, name):
        name_split = name.split('-')
        if name_split[0] != 'meta':
            prefix, world, level = name_split[:3]
            author, prefix = prefix.split('/')
            suffix = '-'.join(name_split[3:])
            self._env_name = '/'.join([author, '-'.join(['meta', prefix, suffix])])
            self._env_level = (int(world) - 1) * 4 + int(level) - 1
        else:
            self._env_name = name
            self._env_level = None
        env = gym.make(self._env_name)
        # modewrapper = wrappers.SetPlayingMode('algo')
        return GymNintendoWrapper(env)

    def _set_info(self, info):
        self.info = copy.copy(info)

    def _action(self, action):
        o, r, is_over, info = self._gym.step(action)
        is_over = info.get('iteration', -1) > self._cur_iter
        if self._env_level is not None:
            if 'distance' in self.info and 'distance' in info:
                r = info['distance'] - self.info['distance']
            else:
                r = 0
        self._set_info(info)
        self._set_current_state(o)
        return r, is_over

    def _restart(self):
        if self._cur_iter < 0:
            self._gym.reset()  # hard mario fceux reset
            if self._env_level is not None:
                self._gym.unwrapped.locked_levels = [False, ] * 32
        else:
            o, _, _, info = self._gym.step(7)  # take one step right
            self._gym.unwrapped.change_level(self._env_level)
        # https://github.com/ppaquette/gym-super-mario/issues/4
        # https://github.com/pathak22/noreward-rl/blob/master/src/env_wrapper.py#L142
        o, _, _, info = self._gym.step(7)  # take right once to start game
        if info.get('ignore', False):  # assuming this happens only in beginning
            self._cur_iter = -1
            self._gym.close()
            self._restart()
        self._cur_iter = info.get('iteration', -1)
        self._set_info(info)
        self._set_current_state(o)

    def _finish(self):
        pass

    def close(self):
        self._gym.close()
