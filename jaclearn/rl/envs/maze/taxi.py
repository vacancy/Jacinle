#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : taxi.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np

from .maze import MazeEnv, CustomLavaWorldEnv
from ...env import SimpleRLEnvBase

__all__ = ['CustomTaxiEnv', 'CustomLavaWorldTaxiEnv']


class CustomTaxiEnv(SimpleRLEnvBase):
    _start_point = None
    _final_point1 = None
    _final_point2 = None

    def __init__(self, maze_env=None, use_coord=False, *args, **kwargs):
        """
        Initialize the maze environment.

        Args:
            self: (todo): write your description
            maze_env: (todo): write your description
            use_coord: (bool): write your description
        """
        super().__init__()
        if maze_env is None:
            maze_env = MazeEnv(*args, **kwargs)
        assert isinstance(maze_env, MazeEnv)
        self._maze_env = maze_env
        self._use_coord = use_coord
        self._phase = 0

    @property
    def maze_env(self):
        """
        : return value of the environment.

        Args:
            self: (todo): write your description
        """
        return self._maze_env

    @property
    def phase(self):
        """
        Returns the phase.

        Args:
            self: (todo): write your description
        """
        return self._phase

    def _get_action_space(self):
        """
        Get action space.

        Args:
            self: (todo): write your description
        """
        return self._maze_env.action_space

    def _refresh_current_state(self):
        """
        Refresh the current state.

        Args:
            self: (todo): write your description
        """
        state = self._maze_env.current_state

        if not self._use_coord:
            self._set_current_state(state)
        else:
            x = np.zeros(shape=(self._maze_env.map_size[1], ), dtype='uint8')
            y = np.zeros(shape=(self._maze_env.map_size[0], ), dtype='uint8')
            if self._phase == 2:
                state = state.copy()
                fp = self._maze_env.final_point
                x[fp[1]], y[fp[0]] = 1, 1
                if (state[fp[0]+1, fp[1]+1] == self._maze_env._colors[3]).all():
                    state[fp[0]+1, fp[1]+1] = self._maze_env._colors[0]
            self._set_current_state((state, x, y))

    def restart(self, start_point=None, final_point1=None, final_point2=None):
        """
        Restart the pipeline.

        Args:
            self: (todo): write your description
            start_point: (str): write your description
            final_point1: (str): write your description
            final_point2: (bool): write your description
        """
        self._start_point = start_point
        self._final_point1 = final_point1
        self._final_point2 = final_point2
        self._maze_env.restart(start_point=start_point, final_point=final_point1)
        self._phase = 1
        self._refresh_current_state()

        super().restart()

    def _restart(self):
        """
        Restart the daemon.

        Args:
            self: (todo): write your description
        """
        pass

    def _action(self, action):
        """
        Return the reward reward.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        # Sanity phase check
        assert self._phase in (1, 2), 'Invalid phase'

        reward, is_over = self._maze_env.action(action)
        if is_over:
            if self._phase == 1:
                self._phase = 2
                self._enter_phase2()
                is_over = False

        self._refresh_current_state()
        return reward, is_over

    def _enter_phase2(self):
        """
        Restart phase.

        Args:
            self: (todo): write your description
        """
        self._maze_env.restart(
            obstacles=self._maze_env.obstacles,
            start_point=self._maze_env.current_point,
            final_point=self._final_point2
        )

    def _finish(self):
        """
        Called bytestring.

        Args:
            self: (todo): write your description
        """
        self._maze_env.finish()


class CustomLavaWorldTaxiEnv(CustomTaxiEnv):
    def __init__(self, maze_env=None, use_coord=False, *args, **kwargs):
        """
        Initialize the environment.

        Args:
            self: (todo): write your description
            maze_env: (todo): write your description
            use_coord: (bool): write your description
        """
        if maze_env is None:
            maze_env = CustomLavaWorldEnv(*args, **kwargs)
        assert isinstance(maze_env, CustomLavaWorldEnv)
        super().__init__(maze_env, use_coord=use_coord)

    def _enter_phase2(self):
        """
        Restart phase.

        Args:
            self: (todo): write your description
        """
        self._maze_env.restart(start_point=self._maze_env.current_point, final_point=self._final_point2)

    def _finish(self):
        """
        Finishes the current phase.

        Args:
            self: (todo): write your description
        """
        super()._finish()

        if self._phase == 2 and self._maze_env._current_point == self._maze_env._final_point:
            self.append_stat('success', 1)
        else:
            self.append_stat('success', 0)
