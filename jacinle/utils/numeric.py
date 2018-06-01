#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : numeric.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import math
from functools import reduce

__all__ = ['safe_sum', 'mean', 'std', 'rms', 'prod', 'divup']


def safe_sum(*values):
    if len(values) == 0:
        return 0
    if len(values) == 1:
        if isinstance(values[0], (tuple, list)):
            return safe_sum(*values[0])
    res = values[0]
    for v in values[1:]:
        res = res + v
    return res


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


def prod(values, default=1):
    if len(values) == 0:
        return default
    return reduce(lambda x, y: x * y, values)


def divup(n, d):
    return n // d + int((n % d) != 0)
