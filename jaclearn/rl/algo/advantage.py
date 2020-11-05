#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : advantage.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from .math import discount_cumsum, compute_gae

__all__ = ['AdvantageComputerBase', 'DiscountedAdvantageComputer', 'GAEComputer']


class AdvantageComputerBase(object):
    def __call__(self, data):
        """
        Calls the call.

        Args:
            self: (todo): write your description
            data: (todo): write your description
        """
        self._compute(data)

    def _compute(self, data):
        """
        Compute the data.

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        raise NotImplementedError()


class DiscountedAdvantageComputer(AdvantageComputerBase):
    def __init__(self, gamma):
        """
        Initialize the gamma.

        Args:
            self: (todo): write your description
            gamma: (float): write your description
        """
        self._gamma = gamma

    def _compute(self, data):
        """
        Compute the total discount.

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        return_ = discount_cumsum(data['reward'], self._gamma)
        advantage = return_ - data['value']

        data['return_'] = return_
        data['advantage'] = advantage


class GAEComputer(AdvantageComputerBase):
    def __init__(self, gamma, lambda_):
        """
        Initialize the gamma.

        Args:
            self: (todo): write your description
            gamma: (float): write your description
            lambda_: (float): write your description
        """
        self._gamma = gamma
        self._lambda = lambda_

    def _compute(self, data):
        """
        Calculate and return value of a given data.

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        return_ = discount_cumsum(data['reward'], self._gamma)
        advantage = compute_gae(data['reward'], data['value'], 0, self._gamma, self._lambda)

        data['return_'] = return_
        data['advantage'] = advantage
