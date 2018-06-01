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
        return getattr(remove_proxies(self), name)


class AutoRestartProxy(ProxyRLEnvBase):
    def _action(self, action):
        r, is_over = self.proxy.action(action)
        if is_over:
            self.finish()
            self.restart()
        return r, is_over


class RepeatActionProxy(ProxyRLEnvBase):
    def __init__(self, other, repeat):
        super().__init__(other)
        self._repeat = repeat

    def _action(self, action):
        total_r = 0
        for i in range(self._repeat):
            r, is_over = self.proxy.action(action)
            total_r += r
            if is_over:
                break
        return total_r, is_over


class NOPFillProxy(ProxyRLEnvBase):
    def __init__(self, other, nr_fill, nop=0):
        super().__init__(other)
        self._nr_fill = nr_fill
        self._nop = nop

    def _action(self, action):
        total_r, is_over = self.proxy.action(action)
        for i in range(self._nr_fill):
            r, is_over = self.proxy.action(self._nop)
            total_r += r
            if is_over:
                break
        return total_r, is_over


class LimitLengthProxy(ProxyRLEnvBase):
    def __init__(self, other, limit):
        super().__init__(other)
        self._limit = limit
        self._cnt = 0

    @property
    def limit(self):
        return self._limit

    def set_limit(self, limit):
        self._limit = limit
        return self

    def _action(self, action):
        r, is_over = self.proxy.action(action)
        self._cnt += 1
        if self._limit is not None and self._cnt >= self._limit:
            is_over = True
        return r, is_over

    def _restart(self, *args, **kwargs):
        super()._restart(*args, **kwargs)
        self._cnt = 0


class MapStateProxy(ProxyRLEnvBase):
    def __init__(self, other, func):
        super().__init__(other)
        self._func = func

    def _get_current_state(self):
        return self._func(self.proxy.current_state)


class MapActionProxy(ProxyRLEnvBase):
    def __init__(self, other, mapping):
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
        return self._action_space

    def _action(self, action):
        assert action < len(self._mapping)
        return self.proxy.action(self._mapping[action])


class HistoryFrameProxy(ProxyRLEnvBase):
    @staticmethod
    def __zeros_like(v):
        if type(v) is tuple:
            return tuple(HistoryFrameProxy.__zeros_like(i) for i in v)
        assert isinstance(v, np.ndarray)
        return np.zeros_like(v, dtype=v.dtype)

    @staticmethod
    def __concat(history):
        last = history[-1]
        if type(last) is tuple:
            return tuple(HistoryFrameProxy.__concat(i) for i in zip(*history))
        return np.concatenate(history, axis=-1)

    def __init__(self, other, history_length):
        super().__init__(other)
        self._history = collections.deque(maxlen=history_length)

    def _get_current_state(self):
        while len(self._history) != self._history.maxlen:
            assert len(self._history) > 0
            v = self._history[-1]
            self._history.appendleft(self.__zeros_like(v))
        return self.__concat(self._history)

    def _set_current_state(self, state):
        if len(self._history) == self._history.maxlen:
            self._history.popleft()
        self._history.append(state)

    # Use shallow copy
    def copy_history(self):
        return copy.copy(self._history)

    def restore_history(self, history):
        assert isinstance(history, collections.deque)
        assert history.maxlen == self._history.maxlen
        self._history = copy.copy(history)

    def _action(self, action):
        r, is_over = self.proxy.action(action)
        self._set_current_state(self.proxy.current_state)
        return r, is_over

    def _restart(self, *args, **kwargs):
        super()._restart(*args, **kwargs)
        self._history.clear()
        self._set_current_state(self.proxy.current_state)


def manipulate_reward(player, func):
    old_func = player._action

    @functools.wraps(old_func)
    def new_func(action):
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
    while not isinstance(environ, proxy_cls) and isinstance(environ, ProxyRLEnvBase):
        environ = environ.proxy
    if isinstance(environ, proxy_cls):
        return environ
    return None
