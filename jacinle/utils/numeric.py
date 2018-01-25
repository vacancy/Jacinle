# -*- coding: utf-8 -*-
# File   : numeric.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.

import math
from functools import reduce

__all__ = ['mean', 'std', 'rms', 'prod', 'divup']


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
