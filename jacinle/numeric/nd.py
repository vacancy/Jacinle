# -*- coding: utf-8 -*-
# File   : nd.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/30/17
#
# This file is part of Jacinle.

import collections
import numpy as np

__all__ = [
    'isndarray',  'is_ndarray',
    'nd_tobatch',
    'nd_concat', 'nd_len', 'nd_batch_size',
    'nd_split_n', 'size_split_n', 'index_select_batch'
]


def isndarray(arr):
    return isinstance(arr, np.ndarray)


# MJY(20170820):: more pythonic naming
is_ndarray = isndarray


def nd_concat(lst):
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0]
    else:
        return np.concatenate(lst)


def nd_len(arr):
    if type(arr) in (int, float):
        return 1
    if isndarray(arr):
        return arr.shape[0]
    return len(arr)


def nd_batch_size(thing):
    if type(thing) in (tuple, list):
        return nd_len(thing[0])
    elif type(thing) in (dict, collections.OrderedDict):
        return nd_len(next(thing.values()))
    else:
        return nd_len(thing)


def size_split_n(full_size, n):
    if full_size is None:
        return None
    result = [full_size // n] * n
    rest = full_size % n
    if rest != 0:
        result[-1] += rest
    return result


def nd_split_n(ndarray, n):
    sub_sizes = size_split_n(len(ndarray), n)
    res = []
    cur = 0
    for i in range(n):
        res.append(ndarray[cur:cur+sub_sizes[i]])
        cur += sub_sizes[i]
    return res


def index_select_batch(data, indices):
    """Gather `indices` as batch indices from `data`, which can either be typical nd array or a
    list of nd array"""
    assert isinstance(indices, (tuple, list)) or (isndarray(indices) and len(indices.shape) == 1)

    if isndarray(data):
        return data[indices]

    assert len(data) > 0 and len(indices) > 0

    sample = np.array(data[0])  # Try to convert the first element to a typical nd array.
    output = np.empty((len(indices), ) + sample.shape, dtype=sample.dtype)
    for i, j in enumerate(indices):
        output[i] = data[j]
    return output
