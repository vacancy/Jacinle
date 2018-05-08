#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : move_right.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np

from ...env import SimpleRLEnvBase
from ...space import DiscreteActionSpace

__all__ = ['MoveRightEnv']


class MoveRightEnv(SimpleRLEnvBase):
    def __init__(self, size=10):
        super().__init__()
        self._size = size
        self._pos = None

    def _get_action_space(self):
        return DiscreteActionSpace(2, ['MOVE_LEFT', 'MOVE_RIGHT'])

    def _get_current_state(self):
        assert 0 <= self._pos < self._size

        state = np.zeros(self._size)
        state[self._pos] = 1
        return state

    def _action(self, action):
        if action == 1:
            self._pos += 1
            reward = 1
        else:
            self._pos = max(0, self._pos - 1)
            reward = -1
        is_over = self._pos >= self._size - 1
        return reward, is_over

    def _restart(self, pos=0, *args, **kwargs):
        self._pos = pos

    def _finish(self, *args, **kwargs):
        pass