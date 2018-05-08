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
        self._compute(data)

    def _compute(self, data):
        raise NotImplementedError()


class DiscountedAdvantageComputer(AdvantageComputerBase):
    def __init__(self, gamma):
        self._gamma = gamma

    def _compute(self, data):
        return_ = discount_cumsum(data['reward'], self._gamma)
        advantage = return_ - data['value']

        data['return_'] = return_
        data['advantage'] = advantage


class GAEComputer(AdvantageComputerBase):
    def __init__(self, gamma, lambda_):
        self._gamma = gamma
        self._lambda = lambda_

    def _compute(self, data):
        return_ = discount_cumsum(data['reward'], self._gamma)
        advantage = compute_gae(data['reward'], data['value'], 0, self._gamma, self._lambda)

        data['return_'] = return_
        data['advantage'] = advantage
