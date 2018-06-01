#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections

from jacinle.utils.cache import cached_property

__all__ = ['RLEnvBase', 'SimpleRLEnvBase', 'ProxyRLEnvBase']


class RLEnvBase(object):
    def __init__(self):
        self._stats = collections.defaultdict(list)

    @property
    def stats(self):
        return self._stats

    def append_stat(self, name, value):
        self._stats[name].append(value)
        return self

    def clear_stats(self):
        self._stats = collections.defaultdict(list)
        return self

    @cached_property
    def action_space(self):
        return self._get_action_space()

    @property
    def current_state(self):
        return self._get_current_state()

    def action(self, action):
        return self._action(action)

    def restart(self, *args, **kwargs):
        return self._restart(*args, **kwargs)

    def finish(self, *args, **kwargs):
        return self._finish(*args, **kwargs)

    def play_one_episode(self, func, ret_states=False, ret_actions=False, restart_kwargs=None, finish_kwargs=None):
        states = []
        actions = []

        self.restart(**(restart_kwargs or {}))
        while True:
            state = self.current_state
            action = func(state)
            r, is_over = self.action(action)
            if ret_actions:
                actions.append(action)
            if ret_states:
                states.append(state)
            if is_over:
                self.finish(**(finish_kwargs or {}))
                break

        if ret_states:
            states.append(self.current_state)

        returns = []
        if ret_states:
            returns.append(states)
        if ret_actions:
            returns.append(actions)
        return returns[0] if len(returns) == 1 else tuple(returns)

    def _get_action_space(self):
        return None

    def _get_current_state(self):
        return None

    def _action(self, action):
        raise NotImplementedError()

    def _restart(self, *args, **kwargs):
        raise NotImplementedError()

    def _finish(self, *args, **kwargs):
        pass

    @property
    def unwrapped(self):
        return self

    def evaluate_one_episode(self, func):
        self.play_one_episode(func)
        score = self.stats['score'][-1]
        self.clear_stats()
        return score


class SimpleRLEnvBase(RLEnvBase):
    _current_state = None

    def __init__(self):
        super().__init__()
        self._reward_history = []

    def _get_current_state(self):
        return self._current_state

    def _set_current_state(self, state):
        self._current_state = state

    def action(self, action):
        r, is_over = self._action(action)
        self._reward_history.append(r)
        return r, is_over

    def restart(self, *args, **kwargs):
        rc = self._restart(*args, **kwargs)
        self._reward_history = []
        return rc

    def finish(self, *args, **kwargs):
        rc = self._finish(*args, **kwargs)
        self.append_stat('score', sum(self._reward_history))
        self.append_stat('length', len(self._reward_history))
        return rc


class ProxyRLEnvBase(RLEnvBase):
    def __init__(self, other):
        super().__init__()
        self.__proxy = other

    @property
    def proxy(self):
        return self.__proxy

    @property
    def stats(self):
        return self.__proxy.stats

    def append_stat(self, name, value):
        self.__proxy.append_stat(name)
        return self

    def clear_stats(self):
        self.__proxy.clear_stats()
        return self

    def _get_action_space(self):
        return self.__proxy.action_space

    def _get_current_state(self):
        return self.__proxy.current_state

    def _action(self, action):
        return self.__proxy.action(action)

    def _restart(self, *args, **kwargs):
        return self.__proxy.restart(*args, **kwargs)

    def _finish(self, *args, **kwargs):
        return self.__proxy.finish(*args, **kwargs)

    @property
    def unwrapped(self):
        return self.proxy.unwrapped
