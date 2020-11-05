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
    """
    Safely convert of values to the sum into a list.

    Args:
        values: (str): write your description
    """
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
    """
    Return the mean of values.

    Args:
        values: (str): write your description
        default: (todo): write your description
    """
    if len(values) == 0:
        return default
    return sum(values) / len(values)


def std(values, default=0):
    """
    Returns the standard deviation.

    Args:
        values: (str): write your description
        default: (todo): write your description
    """
    if len(values) == 0:
        return default
    l = len(values)
    return math.sqrt(sum([v ** 2 for v in values]) / l - (sum(values) / l) ** 2)


def rms(values, default=0):
    """
    Calculates the rms of values.

    Args:
        values: (str): write your description
        default: (todo): write your description
    """
    if len(values) == 0:
        return default
    l = len(values)
    return math.sqrt(sum([v ** 2 for v in values]) / l)


def prod(values, default=1):
    """
    Return the first value of a value. k. a.

    Args:
        values: (str): write your description
        default: (todo): write your description
    """
    if len(values) == 0:
        return default
    return reduce(lambda x, y: x * y, values)


def divup(n, d):
    """
    Divup n times.

    Args:
        n: (int): write your description
        d: (int): write your description
    """
    return n // d + int((n % d) != 0)
