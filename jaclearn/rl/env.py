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
        """
        Initialize the stats.

        Args:
            self: (todo): write your description
        """
        self._stats = collections.defaultdict(list)

    @property
    def stats(self):
        """
        Returns the : class :

        Args:
            self: (todo): write your description
        """
        return self._stats

    def append_stat(self, name, value):
        """
        Add a stat value to the name.

        Args:
            self: (todo): write your description
            name: (str): write your description
            value: (todo): write your description
        """
        self._stats[name].append(value)
        return self

    def clear_stats(self):
        """
        Clears the stats

        Args:
            self: (todo): write your description
        """
        self._stats = collections.defaultdict(list)
        return self

    @cached_property
    def action_space(self):
        """
        Return the action.

        Args:
            self: (todo): write your description
        """
        return self._get_action_space()

    @property
    def current_state(self):
        """
        : returns the current state of the current state.

        Args:
            self: (todo): write your description
        """
        return self._get_current_state()

    def action(self, action):
        """
        Execute an action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        return self._action(action)

    def restart(self, *args, **kwargs):
        """
        Restart the daemon.

        Args:
            self: (todo): write your description
        """
        return self._restart(*args, **kwargs)

    def finish(self, *args, **kwargs):
        """
        Calls the given call to be called arguments.

        Args:
            self: (todo): write your description
        """
        return self._finish(*args, **kwargs)

    def play_one_episode(self, func, ret_states=False, ret_actions=False, restart_kwargs=None, finish_kwargs=None):
        """
        Decorator.

        Args:
            self: (todo): write your description
            func: (todo): write your description
            ret_states: (todo): write your description
            ret_actions: (bool): write your description
            restart_kwargs: (dict): write your description
            finish_kwargs: (dict): write your description
        """
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
        """
        Return the action space.

        Args:
            self: (todo): write your description
        """
        return None

    def _get_current_state(self):
        """
        Get the current state of the current state.

        Args:
            self: (todo): write your description
        """
        return None

    def _action(self, action):
        """
        Execute an action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        raise NotImplementedError()

    def _restart(self, *args, **kwargs):
        """
        Restart the daemon.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()

    def _finish(self, *args, **kwargs):
        """
        Called when the method.

        Args:
            self: (todo): write your description
        """
        pass

    @property
    def unwrapped(self):
        """
        Returns the wrapped wrapped function.

        Args:
            self: (todo): write your description
        """
        return self

    def evaluate_one_episode(self, func):
        """
        Evaluates function.

        Args:
            self: (todo): write your description
            func: (callable): write your description
        """
        self.play_one_episode(func)
        score = self.stats['score'][-1]
        self.clear_stats()
        return score


class SimpleRLEnvBase(RLEnvBase):
    _current_state = None

    def __init__(self):
        """
        Initialize the history.

        Args:
            self: (todo): write your description
        """
        super().__init__()
        self._reward_history = []

    def _get_current_state(self):
        """
        Get the current state

        Args:
            self: (todo): write your description
        """
        return self._current_state

    def _set_current_state(self, state):
        """
        Sets the current state is_state.

        Args:
            self: (todo): write your description
            state: (todo): write your description
        """
        self._current_state = state

    def action(self, action):
        """
        Add an action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        r, is_over = self._action(action)
        self._reward_history.append(r)
        return r, is_over

    def restart(self, *args, **kwargs):
        """
        Restart a new manager.

        Args:
            self: (todo): write your description
        """
        rc = self._restart(*args, **kwargs)
        self._reward_history = []
        return rc

    def finish(self, *args, **kwargs):
        """
        Add the reward.

        Args:
            self: (todo): write your description
        """
        rc = self._finish(*args, **kwargs)
        self.append_stat('score', sum(self._reward_history))
        self.append_stat('length', len(self._reward_history))
        return rc


class ProxyRLEnvBase(RLEnvBase):
    def __init__(self, other):
        """
        Initialize the other

        Args:
            self: (todo): write your description
            other: (todo): write your description
        """
        super().__init__()
        self.__proxy = other

    @property
    def proxy(self):
        """
        Return the proxy

        Args:
            self: (todo): write your description
        """
        return self.__proxy

    @property
    def stats(self):
        """
        Return the stats

        Args:
            self: (todo): write your description
        """
        return self.__proxy.stats

    def append_stat(self, name, value):
        """
        Append a stat.

        Args:
            self: (todo): write your description
            name: (str): write your description
            value: (todo): write your description
        """
        self.__proxy.append_stat(name)
        return self

    def clear_stats(self):
        """
        Clear the stats.

        Args:
            self: (todo): write your description
        """
        self.__proxy.clear_stats()
        return self

    def _get_action_space(self):
        """
        Return the action space.

        Args:
            self: (todo): write your description
        """
        return self.__proxy.action_space

    def _get_current_state(self):
        """
        Get current state

        Args:
            self: (todo): write your description
        """
        return self.__proxy.current_state

    def _action(self, action):
        """
        Execute the action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        return self.__proxy.action(action)

    def _restart(self, *args, **kwargs):
        """
        Restart the proxy.

        Args:
            self: (todo): write your description
        """
        return self.__proxy.restart(*args, **kwargs)

    def _finish(self, *args, **kwargs):
        """
        Calls the proxy.

        Args:
            self: (todo): write your description
        """
        return self.__proxy.finish(*args, **kwargs)

    @property
    def unwrapped(self):
        """
        Unwraps the wrapped proxy.

        Args:
            self: (todo): write your description
        """
        return self.proxy.unwrapped
