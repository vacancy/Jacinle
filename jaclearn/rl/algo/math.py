#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : math.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import scipy
import scipy.signal
import numpy as np


def discount_cumsum(x, gamma):
    """Compute the discounted cumulative summation of an 1-d array.
    From https://github.com/rll/rllab/blob/master/rllab/misc/special.py"""
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def discount_return(x, discount):
    """Compute the discounted return summation of an 1-d array.
    From https://github.com/rll/rllab/blob/master/rllab/misc/special.py"""
    return np.sum(x * (discount ** np.arange(len(x))))


def normalize_advantage(adv):
    """
    Normalize the standard deviation.

    Args:
        adv: (todo): write your description
    """
    return (adv - adv.mean()) / adv.std()


class ObservationNormalizer(object):
    _eps = 1e-6

    """Normalize the input with a moving average."""
    def __init__(self, filter_mean=True):
        """
        Initialize thread

        Args:
            self: (todo): write your description
            filter_mean: (todo): write your description
        """
        self.m1 = 0
        self.v = 0
        self.std = 0
        self.n = 0.
        self.filter_mean = filter_mean

        import threading
        self.lock = threading.Lock()

    def __call__(self, o):
        """
        Calls self.

        Args:
            self: (todo): write your description
            o: (array): write your description
        """
        with self.lock:
            return self.normalize(o)

    def normalize(self, o):
        """
        Normalize o and v ).

        Args:
            self: (todo): write your description
            o: (todo): write your description
        """
        self.m1 = self.m1 * (self.n / (self.n + 1)) + o * 1 / (1 + self.n)
        self.v = self.v * (self.n / (self.n + 1)) + (o - self.m1) ** 2 * 1 / (1 + self.n)
        self.std = (self.v + self._eps) ** .5  # std
        self.n += 1
        if self.filter_mean:
            o1 = (o - self.m1) / self.std
        else:
            o1 = o / self.std
        o1 = (o1 > 10) * 10 + (o1 < -10) * (-10) + (o1 < 10) * (o1 > -10) * o1
        return o1


class LinearValueRegressor(object):
    _name = 'linear_value_regressor'
    coeffs = None

    def _features(self, states, steps):
        """
        Concatenate features.

        Args:
            self: (todo): write your description
            states: (array): write your description
            steps: (array): write your description
        """
        o = states.astype('float32').reshape(states.shape[0], -1)
        s = steps.reshape(steps.shape[0], -1) / 100.
        return np.concatenate([o, s ** 2, s, s ** 2, np.ones((states.shape[0], 1))], axis=1)

    def fit(self, states, steps, returns):
        """
        Fits the model.

        Args:
            self: (todo): write your description
            states: (array): write your description
            steps: (array): write your description
            returns: (todo): write your description
        """
        featmat = self._features(states, steps)
        n_col = featmat.shape[1]
        lamb = 2.0
        self.coeffs = np.linalg.lstsq(featmat.T.dot(featmat) + lamb * np.identity(n_col), featmat.T.dot(returns))[0]

    def predict(self, states, steps):
        """
        Predict the state at states.

        Args:
            self: (array): write your description
            states: (list): write your description
            steps: (array): write your description
        """
        if self.coeffs is None:
            return np.zeros(states.shape[0])

        return self._features(states, steps).dot(self.coeffs)

    def register_snapshot_parts(self, env):
        """
        Register a snapshot to the snapshot.

        Args:
            self: (todo): write your description
            env: (todo): write your description
        """
        env.add_snapshot_part(self._name, self._dump_params, self._load_params)

    def _dump_params(self):
        """
        Return a list of this class.

        Args:
            self: (todo): write your description
        """
        return self.coeffs

    def _load_params(self, coeffs):
        """
        Load parameters from a file.

        Args:
            self: (todo): write your description
            coeffs: (int): write your description
        """
        self.coeffs = coeffs


def compute_gae(rewards, values, next_val, gamma, lambda_):
    """
    R calculate the smoothing of a sample.

    Args:
        rewards: (todo): write your description
        values: (str): write your description
        next_val: (str): write your description
        gamma: (todo): write your description
        lambda_: (todo): write your description
    """
    assert len(rewards) == len(values)
    size = len(rewards)
    adv_batch = np.empty((size, ), dtype='float32')

    td_i = rewards[size - 1] + gamma * next_val - values[size - 1]
    adv_batch[size - 1] = td_i

    for i in range(size - 2, -1, -1):
        td_i = rewards[i] + gamma * values[i+1] - values[i] + gamma * lambda_ * td_i
        adv_batch[i] = td_i

    return adv_batch
