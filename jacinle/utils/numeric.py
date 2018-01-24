# -*- coding: utf-8 -*-
# File   : numeric.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.

import math

__all__ = ['mean', 'std', 'rms']


def mean(values, default=0):
    if len(values) == 0:
        return default
    return sum(values) / len(values)


def std(values, default=0):
    if len(values) == 0:
        return default
    l = len(values)
    return math.sqrt(sum([v ** 2 for v in values]) / l - (sum(values) / l) ** 2)


def rms(values, default=0):
    if len(values) == 0:
        return default
    l = len(values)
    return math.sqrt(sum([v ** 2 for v in values]) / l)
