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
        """
        Initialize the action.

        Args:
            self: (todo): write your description
            action_meanings: (todo): write your description
        """
        self.__rng = random.gen_rng()
        self._action_meanings = action_meanings

    @property
    def rng(self):
        """
        Return the rng

        Args:
            self: (todo): write your description
        """
        return self.__rng

    @property
    def action_meanings(self):
        """
        Returns the action action.

        Args:
            self: (todo): write your description
        """
        return self._action_meanings

    def sample(self, theta=None):
        """
        Return a sample

        Args:
            self: (todo): write your description
            theta: (float): write your description
        """
        return self._sample(theta)

    def _sample(self, theta=None):
        """
        Return a sample from the sample.

        Args:
            self: (todo): write your description
            theta: (float): write your description
        """
        return None


class DiscreteActionSpace(ActionSpaceBase):
    def __init__(self, nr_actions, action_meanings=None):
        """
        Initialize actions.

        Args:
            self: (todo): write your description
            nr_actions: (todo): write your description
            action_meanings: (todo): write your description
        """
        super().__init__(action_meanings=action_meanings)
        self._nr_actions = nr_actions

    @property
    def nr_actions(self):
        """
        Return a list of actions.

        Args:
            self: (todo): write your description
        """
        return self._nr_actions

    def _sample(self, theta=None):
        """
        Return a copy of the random unit.

        Args:
            self: (todo): write your description
            theta: (float): write your description
        """
        if theta is None:
            return self.rng.choice(self._nr_actions)
        return self.rng.choice(self._nr_actions, p=theta)


class ContinuousActionSpace(ActionSpaceBase):
    @staticmethod
    def __canonize_bound(v, shape):
        """
        Return the bounding box of v.

        Args:
            v: (todo): write your description
            shape: (int): write your description
        """
        if type(v) is np.ndarray:
            assert v.shape == shape, 'Invalid shape for bound value: expect {}, got {}.'.format(
                    shape, v.shape)
            return v

        assert type(v) in (int, float), 'Invalid type for bound value.'
        return np.ones(shape=shape, dtype='float32') * v

    def __init__(self, low, high=None, shape=None, action_meanings=None):
        """
        Initialize the bounding box.

        Args:
            self: (todo): write your description
            low: (todo): write your description
            high: (int): write your description
            shape: (int): write your description
            action_meanings: (todo): write your description
        """
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
        """
        Low - low - level.

        Args:
            self: (todo): write your description
        """
        return self._low

    @property
    def high(self):
        """
        High level of the high high high high level.

        Args:
            self: (todo): write your description
        """
        return self._high

    @property
    def shape(self):
        """
        Returns the shape of the shape.

        Args:
            self: (todo): write your description
        """
        return self._shape

    def _sample(self, theta=None):
        """
        Sample from the distribution.

        Args:
            self: (todo): write your description
            theta: (float): write your description
        """
        if theta is not None:
            mu, std = theta
            return self.rng.randn(*self.shape) * std + mu
        return self.rng.uniform(self._low, self._high)