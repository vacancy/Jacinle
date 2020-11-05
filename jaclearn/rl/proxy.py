#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : proxy.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import copy
import functools
import collections
import numpy as np

from .env import ProxyRLEnvBase
from .space import DiscreteActionSpace

__all__ = [
    'TransparentAttributeProxy',
    'AutoRestartProxy',
    'RepeatActionProxy',
    'NOPFillProxy',
    'LimitLengthProxy',
    'MapStateProxy',
    'MapActionProxy',
    'HistoryFrameProxy',
    'manipulate_reward',
    'remove_proxies',
    'find_proxy'
]


class TransparentAttributeProxy(ProxyRLEnvBase):
    def __getattr__(self, name):
        """
        Get a named attribute by name.

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        return getattr(remove_proxies(self), name)


class AutoRestartProxy(ProxyRLEnvBase):
    def _action(self, action):
        """
        Restart an action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        r, is_over = self.proxy.action(action)
        if is_over:
            self.finish()
            self.restart()
        return r, is_over


class RepeatActionProxy(ProxyRLEnvBase):
    def __init__(self, other, repeat):
        """
        Initialize the other.

        Args:
            self: (todo): write your description
            other: (todo): write your description
            repeat: (int): write your description
        """
        super().__init__(other)
        self._repeat = repeat

    def _action(self, action):
        """
        Perform an action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        total_r = 0
        for i in range(self._repeat):
            r, is_over = self.proxy.action(action)
            total_r += r
            if is_over:
                break
        return total_r, is_over


class NOPFillProxy(ProxyRLEnvBase):
    def __init__(self, other, nr_fill, nop=0):
        """
        Wrapper for nanop.

        Args:
            self: (todo): write your description
            other: (todo): write your description
            nr_fill: (str): write your description
            nop: (todo): write your description
        """
        super().__init__(other)
        self._nr_fill = nr_fill
        self._nop = nop

    def _action(self, action):
        """
        Return a new action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        total_r, is_over = self.proxy.action(action)
        for i in range(self._nr_fill):
            r, is_over = self.proxy.action(self._nop)
            total_r += r
            if is_over:
                break
        return total_r, is_over


class LimitLengthProxy(ProxyRLEnvBase):
    def __init__(self, other, limit):
        """
        Initialize the current limit.

        Args:
            self: (todo): write your description
            other: (todo): write your description
            limit: (int): write your description
        """
        super().__init__(other)
        self._limit = limit
        self._cnt = 0

    @property
    def limit(self):
        """
        Returns the number of results.

        Args:
            self: (todo): write your description
        """
        return self._limit

    def set_limit(self, limit):
        """
        Sets the limit : paramater

        Args:
            self: (todo): write your description
            limit: (int): write your description
        """
        self._limit = limit
        return self

    def _action(self, action):
        """
        Return a tuple.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        r, is_over = self.proxy.action(action)
        self._cnt += 1
        if self._limit is not None and self._cnt >= self._limit:
            is_over = True
        return r, is_over

    def _restart(self, *args, **kwargs):
        """
        Restart the thread.

        Args:
            self: (todo): write your description
        """
        super()._restart(*args, **kwargs)
        self._cnt = 0


class MapStateProxy(ProxyRLEnvBase):
    def __init__(self, other, func):
        """
        Initialize self. __init__.

        Args:
            self: (todo): write your description
            other: (todo): write your description
            func: (callable): write your description
        """
        super().__init__(other)
        self._func = func

    def _get_current_state(self):
        """
        Returns the current state of the current state.

        Args:
            self: (todo): write your description
        """
        return self._func(self.proxy.current_state)


class MapActionProxy(ProxyRLEnvBase):
    def __init__(self, other, mapping):
        """
        Initialize the mapping.

        Args:
            self: (todo): write your description
            other: (todo): write your description
            mapping: (dict): write your description
        """
        super().__init__(other)
        assert type(mapping) in [tuple, list]
        for i in mapping:
            assert type(i) is int
        self._mapping = mapping
        action_space = other.action_space
        assert isinstance(action_space, DiscreteActionSpace)
        action_meanings = [action_space.action_meanings[i] for i in mapping]
        self._action_space = DiscreteActionSpace(len(mapping), action_meanings)

    def _get_action_space(self):
        """
        Get the action space.

        Args:
            self: (todo): write your description
        """
        return self._action_space

    def _action(self, action):
        """
        Perform action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        assert action < len(self._mapping)
        return self.proxy.action(self._mapping[action])


class HistoryFrameProxy(ProxyRLEnvBase):
    @staticmethod
    def __zeros_like(v):
        """
        Return an numpy. ndarray.

        Args:
            v: (array): write your description
        """
        if type(v) is tuple:
            return tuple(HistoryFrameProxy.__zeros_like(i) for i in v)
        assert isinstance(v, np.ndarray)
        return np.zeros_like(v, dtype=v.dtype)

    @staticmethod
    def __concat(history):
        """
        Concatenate the concatenation of a given history.

        Args:
            history: (todo): write your description
        """
        last = history[-1]
        if type(last) is tuple:
            return tuple(HistoryFrameProxy.__concat(i) for i in zip(*history))
        return np.concatenate(history, axis=-1)

    def __init__(self, other, history_length):
        """
        Initialize the history.

        Args:
            self: (todo): write your description
            other: (todo): write your description
            history_length: (int): write your description
        """
        super().__init__(other)
        self._history = collections.deque(maxlen=history_length)

    def _get_current_state(self):
        """
        Returns the current state of the history.

        Args:
            self: (todo): write your description
        """
        while len(self._history) != self._history.maxlen:
            assert len(self._history) > 0
            v = self._history[-1]
            self._history.appendleft(self.__zeros_like(v))
        return self.__concat(self._history)

    def _set_current_state(self, state):
        """
        Sets the current state.

        Args:
            self: (todo): write your description
            state: (todo): write your description
        """
        if len(self._history) == self._history.maxlen:
            self._history.popleft()
        self._history.append(state)

    # Use shallow copy
    def copy_history(self):
        """
        Returns a copy of this instance.

        Args:
            self: (todo): write your description
        """
        return copy.copy(self._history)

    def restore_history(self, history):
        """
        Restore the history of the history.

        Args:
            self: (todo): write your description
            history: (todo): write your description
        """
        assert isinstance(history, collections.deque)
        assert history.maxlen == self._history.maxlen
        self._history = copy.copy(history)

    def _action(self, action):
        """
        Return the action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        r, is_over = self.proxy.action(action)
        self._set_current_state(self.proxy.current_state)
        return r, is_over

    def _restart(self, *args, **kwargs):
        """
        Restart the proxy.

        Args:
            self: (todo): write your description
        """
        super()._restart(*args, **kwargs)
        self._history.clear()
        self._set_current_state(self.proxy.current_state)


def manipulate_reward(player, func):
    """
    Decorator to specify a new reward.

    Args:
        player: (todo): write your description
        func: (todo): write your description
    """
    old_func = player._action

    @functools.wraps(old_func)
    def new_func(action):
        """
        Return a new function that can be used function.

        Args:
            action: (str): write your description
        """
        r, is_over = old_func(action)
        return func(r), is_over

    player._action = new_func
    return player


def remove_proxies(environ):
    """Remove all wrapped proxy environs"""
    while isinstance(environ, ProxyRLEnvBase):
        environ = environ.proxy
    return environ


def find_proxy(environ, proxy_cls):
    """
    Finds a proxy.

    Args:
        environ: (dict): write your description
        proxy_cls: (todo): write your description
    """
    while not isinstance(environ, proxy_cls) and isinstance(environ, ProxyRLEnvBase):
        environ = environ.proxy
    if isinstance(environ, proxy_cls):
        return environ
    return None
