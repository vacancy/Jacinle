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
from typing import Any, Optional, Union, Sequence, Tuple, Callable

__all__ = [
    'get_2dshape', 'get_3dshape', 'get_4dshape', 'get_nd_shape',
    'astuple', 'asshape',
    'canonize_args_list',
    'UniqueValueGetter'
]


def get_2dshape(x: Optional[Union[int, Sequence[int]]], default: Tuple[int, int] = None, type: type = int) -> Tuple[int, int]:
    """Convert a value or a tuple to a tuple of length 2.

    Args:
        x: a value of type `type`, or a tuple of length 2. If the input is a single value, it will be duplicated to a tuple of length 2.
        default: default value.
        type: expected type of the element.

    Returns:
        a tuple of length 2.
    """
    if x is None:
        return default
    if isinstance(x, collections.abc.Sequence):
        x = tuple(x)
        if len(x) == 1:
            return x[0], x[0]
        else:
            assert len(x) == 2, '2dshape must be of length 1 or 2'
            return x
    else:
        x = type(x)
        return x, x


def get_3dshape(x: Optional[Union[int, Sequence[int]]], default: Tuple[int, int, int] = None, type: type = int) -> Tuple[int, int, int]:
    """Convert a value or a tuple to a tuple of length 3.

    Args:
        x: a value of type `type`, or a tuple of length 3. If the input is a single value, it will be duplicated to a tuple of length 3.
        default: default value.
        type: expected type of the element.

    Returns:
        a tuple of length 3.
    """

    if x is None:
        return default
    if isinstance(x, collections.abc.Sequence):
        x = tuple(x)
        if len(x) == 1:
            return x[0], x[0], x[0]
        else:
            assert len(x) == 3, '3dshape must be of length 1 or 3'
            return x
    else:
        x = type(x)
        return x, x, x


def get_4dshape(x: Optional[Union[int, Sequence[int]]], default: Tuple[int, int, int, int] = None, type: type = int) -> Tuple[int, int, int, int]:
    """Convert a value or a tuple to a tuple of length 4.

    Args:
        x: a value of type `type`, or a tuple of length 4. If there is only one value, it will return (1, x, x, 1).
            If there are two values, it will return (1, x[0], x[1], 1).
        default: default value.
        type: expected type of the element.

    Returns:
        a tuple of length 4.
    """
    if x is None:
        return default
    if isinstance(x, collections.abc.Sequence):
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


def get_nd_shape(x: Optional[Union[int, Sequence[int]]], ndim: int, default: Tuple[int, ...] = None, type: type = int) -> Tuple[int, ...]:
    """Convert a value or a tuple to a tuple of length `ndim`.

    Args:
        x: a value of type `type`, or a tuple of length `ndim`. If the input is a single value, it will be duplicated to a tuple of length `ndim`.
        ndim: the expected length of the tuple.
        default: default value.
        type: expected type of the element.

    Returns:
        a tuple of length `ndim`.
    """
    if x is None:
        return default
    if isinstance(x, collections.abc.Sequence):
        x = tuple(x)
        if len(x) == 1:
            return (x[0],) * ndim
        else:
            assert len(x) == ndim, f'{ndim}dshape must be of length 1 or {ndim}'
            return x
    else:
        x = type(x)
        return (x,) * ndim


def astuple(arr_like: Any) -> Tuple:
    """Convert a sequence or a single value to a tuple. This method differ from the system method `tuple` in that
    a single value (incl. int, string, bytes) will be converted to a tuple of length 1.

    Args:
        arr_like: a sequence or a single value.

    Returns:
        a tuple.
    """
    if type(arr_like) is tuple:
        return arr_like
    elif isinstance(arr_like, collections.abc.Sequence) and not isinstance(arr_like, (str, bytes)):
        return tuple(arr_like)
    else:
        return tuple((arr_like,))


def asshape(arr_like: Optional[Union[int, Sequence[int]]]) -> Optional[Tuple[int, ...]]:
    """Convert a sequence or a single value to a tuple of integers. It will return None if the input is None.

    Args:
        arr_like: a sequence or a single value.

    Returns:
        a tuple of integers.
    """
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


def canonize_args_list(args: Tuple[Any], *, allow_empty: bool = False, cvt: Optional[Callable[[Any], Any]] = None) -> Tuple[Any]:
    """Convert the argument list to a tuple of values. This is useful to make unified interface for shape-related operations.

    Example:
        .. code-block:: python

            def foo(*args):
                args = canonize_args_list(args, allow_empty=True)
                print(args)

            foo(1, 2, 3)  # (1, 2, 3)
            foo((1, 2, 3))  # (1, 2, 3)
            foo(1)  # (1,)
            foo()  # ()

    Args:
        args: the argument list.
        allow_empty: whether to allow empty argument list.
        cvt: a function to be applied to each element.
    """

    if not allow_empty and not args:
        raise TypeError('at least one argument must be provided')

    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    if cvt is not None:
        args = tuple(map(cvt, args))
    return args


class UniqueValueGetter(object):
    """A helper class to ensure that a value is unique.

    Example:
        .. code-block:: python

            uvg = UniqueValueGetter()
            uvg.set(1)
            uvg.set(2)  # will raise ValueError
            uvg.set(1)

            print(uvg.get())  # 1
    """

    def __init__(self, msg: str = 'Unique value checking failed', default: Any = None):
        """Initialize the UniqueValueGetter.

        Args:
            msg: the error message.
            default: the default value.
        """
        self._msg = msg
        self._val = None
        self._default = default

    def set(self, v):
        assert self._val is None or self._val == v, self._msg + ': expect={} got={}'.format(self._val, v)
        self._val = v

    def get(self):
        return self._val or self._default

