#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : space.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np

import jacinle.random as random

__all__ = [
    'ActionSpaceBase', 'DiscreteActionSpace', 'ContinuousActionSpace'
]


class ActionSpaceBase(object):
    def __init__(self, action_meanings=None):
        self.__rng = random.gen_rng()
        self._action_meanings = action_meanings

    @property
    def rng(self):
        return self.__rng

    @property
    def action_meanings(self):
        return self._action_meanings

    def sample(self, theta=None):
        return self._sample(theta)

    def _sample(self, theta=None):
        return None


class DiscreteActionSpace(ActionSpaceBase):
    def __init__(self, nr_actions, action_meanings=None):
        super().__init__(action_meanings=action_meanings)
        self._nr_actions = nr_actions

    @property
    def nr_actions(self):
        return self._nr_actions

    def _sample(self, theta=None):
        if theta is None:
            return self.rng.choice(self._nr_actions)
        return self.rng.choice(self._nr_actions, p=theta)


class ContinuousActionSpace(ActionSpaceBase):
    @staticmethod
    def __canonize_bound(v, shape):
        if type(v) is np.ndarray:
            assert v.shape == shape, 'Invalid shape for bound value: expect {}, got {}.'.format(
                    shape, v.shape)
            return v

        assert type(v) in (int, float), 'Invalid type for bound value.'
        return np.ones(shape=shape, dtype='float32') * v

    def __init__(self, low, high=None, shape=None, action_meanings=None):
        super().__init__(action_meanings=action_meanings)

        if high is None:
            low, high = -low, low

        if shape is None:
            assert low is not None and high is not None, 'Must provide low and high.'
            low, high = np.array(low), np.array(high)
            assert low.shape == high.shape, 'Low and high must have same shape, got: {} and {}.'.format(
                    low.shape, high.shape)

            self._low = low
            self._high = high
            self._shape = low.shape
        else:
            self._low = self.__canonize_bound(low, shape)
            self._high = self.__canonize_bound(high, shape)
            self._shape = shape

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def shape(self):
        return self._shape

    def _sample(self, theta=None):
        if theta is not None:
            mu, std = theta
            return self.rng.randn(*self.shape) * std + mu
        return self.rng.uniform(self._low, self._high)