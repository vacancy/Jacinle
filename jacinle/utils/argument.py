#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : argument.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/14/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections

__all__ = [
    'get_2dshape', 'get_3dshape', 'get_4dshape',
    'astuple', 'asshape',
    'canonize_args_list',
    'UniqueValueGetter'
]


def get_2dshape(x, default=None, type=int):
    """Convert a value or a tuple to a tuple of length 2.
    Args:
        x: a value of type `type`, or a tuple of length 2.
        default: default value.
        type: expected type of the element.

    Returns: a tuple of length 2.
    """
    if x is None:
        return default
    if isinstance(x, collections.Sequence):
        x = tuple(x)
        if len(x) == 1:
            return x[0], x[0]
        else:
            assert len(x) == 2, '2dshape must be of length 1 or 2'
            return x
    else:
        x = type(x)
        return x, x


def get_3dshape(x, default=None, type=int):
    if x is None:
        return default
    if isinstance(x, collections.Sequence):
        x = tuple(x)
        if len(x) == 1:
            return x[0], x[0], x[0]
        else:
            assert len(x) == 3, '3dshape must be of length 1 or 3'
            return x
    else:
        x = type(x)
        return x, x, x


def get_4dshape(x, default=None, type=int):
    if x is None:
        return default
    if isinstance(x, collections.Sequence):
        x = tuple(x)
        if len(x) == 1:
            return 1, x[0], x[0], 1
        elif len(x) == 2:
            return 1, x[0], x[1], 1
        else:
            assert len(x) == 4, '4dshape must be of length 1, 2, or 4'
            return x
    else:
        x = type(x)
        return 1, x, x, 1


def astuple(arr_like):
    """Convert a sequence or a single value to a tuple. This method differ from the system method `tuple` in that
    a single value (incl. int, string, bytes) will be converted to a tuple of length 1.
    Args:
        arr_like: a sequence or a single value.

    Returns: a tuple.
    """
    if type(arr_like) is tuple:
        return arr_like
    elif isinstance(arr_like, collections.Sequence) and not isinstance(arr_like, (str, bytes)):
        return tuple(arr_like)
    else:
        return tuple((arr_like,))


def asshape(arr_like):
    if type(arr_like) is tuple:
        return arr_like
    elif type(arr_like) is int:
        if arr_like == 0:
            return tuple()
        else:
            return tuple((arr_like,))
    elif arr_like is None:
        return None,
    else:
        return tuple(arr_like)


def canonize_args_list(args, *, allow_empty=False, cvt=None):
    """Convert the argument list to a tuple of values. This is useful to make unified interface for shape-related
    operations. E.g.,

    >>> np.zeros(6).reshape(2, 3)
    >>> np.zeros(6).reshape((2, 3))

    Args:
        args:
        allow_empty:
        cvt:

    Returns:

    """
    if not allow_empty and not args:
        raise TypeError('at least one argument must be provided')

    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    if cvt is not None:
        args = tuple(map(cvt, args))
    return args


class UniqueValueGetter(object):
    def __init__(self, msg='unique value check failed', default=None):
        self._msg = msg
        self._val = None
        self._default = default

    def set(self, v):
        assert self._val is None or self._val == v, self._msg + ': expect={} got={}'.format(self._val, v)
        self._val = v

    def get(self):
        return self._val or self._default
