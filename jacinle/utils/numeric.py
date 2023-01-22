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
    """A safe sum function that uses the first value as the initial value. It can be used as a replacement of :func:`sum`."""
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
    """A mean function that returns the default value when the input is empty."""
    if len(values) == 0:
        return default
    return sum(values) / len(values)


def std(values, default=0):
    """A standard deviation function that returns the default value when the input is empty."""
    if len(values) == 0:
        return default
    l = len(values)
    return math.sqrt(sum([v ** 2 for v in values]) / l - (sum(values) / l) ** 2)


def rms(values, default=0):
    """A root mean square function that returns the default value when the input is empty."""
    if len(values) == 0:
        return default
    l = len(values)
    return math.sqrt(sum([v ** 2 for v in values]) / l)


def prod(values, default=1):
    """A product function that returns the default value when the input is empty."""
    if len(values) == 0:
        return default
    return reduce(lambda x, y: x * y, values)


def divup(n, d):
    """Divide n by d and round up."""
    return n // d + int((n % d) != 0)
