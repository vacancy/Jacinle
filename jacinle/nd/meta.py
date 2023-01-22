#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : meta.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/30/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections
from typing import Any, Optional, Sequence, List

import numpy as np

__all__ = [
    'isndarray',  'is_ndarray',
    'nd_concat', 'nd_len', 'nd_batch_size',
    'nd_split_n', 'size_split_n'
]


def isndarray(thing: Any) -> bool:
    """Check if the given object is a numpy array."""
    return isinstance(thing, np.ndarray)


def is_ndarray(thing: Any) -> bool:
    return isinstance(thing, np.ndarray)


def nd_concat(list_of_arrays: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Concatenate a list of numpy arrays. This function handles the case when the list is empty or contains only one element.

    Args:
        list_of_arrays: a list of numpy arrays.

    Returns:
        the concatenated array, or None if the list is empty.
    """
    if len(list_of_arrays) == 0:
        return None
    elif len(list_of_arrays) == 1:
        return list_of_arrays[0]
    else:
        return np.concatenate(list_of_arrays)


def nd_len(thing: Any) -> int:
    """Get the length of a numpy array. This function handles the case when the input is a scalar or plain Python objects.

    Args:
        thing: the input array.

    Returns:
        the length of the array, or 1 if the input is a scalar or plain Python objects.
    """
    if type(thing) in (int, float):
        return 1
    if isndarray(thing):
        return thing.shape[0]
    return len(thing)


def nd_batch_size(thing: Any) -> int:
    """Get the batch size of a numpy array. This function handles the case when the input a nested list or dict.

    Examples:
        >>> nd_batch_size(np.array([1, 2, 3]))
        3
        >>> nd_batch_size([np.zeros((2, 3)), np.zeros((2, 5))])
        2
        >>> nd_batch_size({'a': np.zeros((2, 3)), 'b': np.zeros((2, 5))})
        2

    Args:
        thing: the input array or nested list/dict.

    Returns:
        the batch size of the array.
    """
    if type(thing) in (tuple, list):
        return nd_len(thing[0])
    elif type(thing) in (dict, collections.OrderedDict):
        return nd_len(next(thing.values()))
    else:
        return nd_len(thing)


def size_split_n(full_size: Optional[int], n: int) -> Optional[List[int]]:
    """Split a size into n parts. If the size is not divisible by n, the last part will be larger.
    When the size is None, None will be returned.

    Args:
        full_size: the size to be split.
        n: the number of parts.

    Returns:
        a list of sizes.
    """
    if full_size is None:
        return None
    result = [full_size // n] * n
    rest = full_size % n
    if rest != 0:
        result[-1] += rest
    return result


def nd_split_n(ndarray: np.ndarray, n: int) -> List[np.ndarray]:
    """Split a numpy array into n parts. If the size is not divisible by n, the last part will be larger.

    Args:
        ndarray: the array to be split.
        n: the number of parts.

    Returns:
        a list of arrays.
    """
    sub_sizes = size_split_n(len(ndarray), n)
    res = []
    cur = 0
    for i in range(n):
        res.append(ndarray[cur:cur+sub_sizes[i]])
        cur += sub_sizes[i]
    return res
