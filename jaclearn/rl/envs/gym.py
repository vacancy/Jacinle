#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gym.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import threading
import numpy as np
import collections

import jacinle.io as jacio

from ..env import SimpleRLEnvBase, ProxyRLEnvBase
from ..space import DiscreteActionSpace, ContinuousActionSpace

try:
    import gym
    import gym.wrappers
except ImportError:
    gym = None

_ENV_LOCK = threading.Lock()


def get_env_lock():
    return _ENV_LOCK


__all__ = ['GymRLEnv', 'GymAtariRLEnv', 'GymPreventStuckProxy']


class GymRLEnv(SimpleRLEnvBase):
    def __init__(self, name, dump_dir=None, force_dump=False, state_mode='DEFAULT'):
        super().__init__()

        with get_env_lock():
            self._gym = self._make_env(name)

        if dump_dir:
            jacio.mkdir(dump_dir)
            self._gym = gym.wrappers.Monitor(self._gym, dump_dir, force=force_dump)

        assert state_mode in ('DEFAULT', 'RENDER', 'BOTH')
        self._state_mode = state_mode

    def _make_env(self, name):
        return gym.make(name)

    @property
    def gym(self):
        return self._gym

    def render(self, mode='human', close=False):
        return self._gym.render(mode=mode, close=close)

    def _set_current_state(self, o):
        if self._state_mode == 'DEFAULT':
            pass
        else:
            rendered = self.render('rgb_array')
            if self._state_mode == 'RENDER':
                o = rendered
            else:
                o = (o, rendered)
        super()._set_current_state(o)

    def _get_action_space(self):
        spc = self._gym.action_space

        if isinstance(spc, gym.spaces.discrete.Discrete):
            try:
                action_meanings = self._gym.unwrapped.get_action_meanings()
            except AttributeError:
                if 'Atari' in self._gym.unwrapped.__class__.__name__:
                    from gym.envs.atari.atari_env import ACTION_MEANING
                    action_meanings = [ACTION_MEANING[i] for i in range(spc.n)]
                else:
                    action_meanings = ['unknown{}'.format(i) for i in range(spc.n)]
            return DiscreteActionSpace(spc.n, action_meanings=action_meanings)
        elif isinstance(spc, gym.spaces.box.Box):
            return ContinuousActionSpace(spc.low, spc.high, spc.shape)
        else:
            raise ValueError('Unknown gym space spec: {}.'.format(spc))

    def _action(self, action):
        # hack for continuous control
        if type(action) in (tuple, list):
            action = np.array(action)

        o, r, is_over, info = self._gym.step(action)
        self._set_current_state(o)
        return r, is_over

    def _restart(self):
        o = self._gym.reset()
        self._set_current_state(o)

    def _finish(self):
        self._gym.close()


class GymAtariRLEnv(GymRLEnv):
    def __init__(self, name, *args, live_lost_as_eoe=True, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._live_lost_as_eoe = live_lost_as_eoe

    def _action(self, action):
        old_lives = self._gym.unwrapped.ale.lives()
        r, is_over = super()._action(action)
        new_lives = self._gym.unwrapped.ale.lives()
        if self._live_lost_as_eoe and old_lives > new_lives:
            is_over = True
        return r, is_over


class GymPreventStuckProxy(ProxyRLEnvBase):
    def __init__(self, other, max_repeat, action):
        super().__init__(other)
        self._action_list = collections.deque(maxlen=max_repeat)
        self._insert_action = action

    def _action(self, action):
        self._action_list.append(action)
        if self._action_list.count(self._action_list[0]) == self._action_list.maxlen:
            action = self._insert_action
        r, is_over = self.proxy.action(action)
        if is_over:
            self._action_list.clear()
        return r, is_over

    def _restart(self, *args, **kwargs):
        super()._restart(*args, **kwargs)
        self._action_list.clear()
