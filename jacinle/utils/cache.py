#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cache.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import functools
import collections
import os.path as osp
import threading

from jacinle.logging import get_logger
from .meta import synchronized

logger = get_logger(__file__)

__all__ = ['cached_property', 'cached_result', 'fs_cached_result']


class cached_property:
    def __init__(self, fget):
        self.fget = fget
        self.__module__ = fget.__module__
        self.__name__ = fget.__name__
        self.__doc__ = fget.__doc__
        self.__cache_key = '__result_cache_{}_{}'.format(
            fget.__name__, id(fget))
        self.__mutex = collections.defaultdict(threading.Lock)

    def __get__(self, instance, owner):
        with self.__mutex[id(instance)]:
            if instance is None:
                return self.fget
            v = getattr(instance, self.__cache_key, None)
            if v is not None:
                return v
            v = self.fget(instance)
            assert v is not None
            setattr(instance, self.__cache_key, v)
            return v


def cached_result(func):
    def impl():
        nonlocal impl
        ret = func()
        impl = lambda: ret
        return ret

    @synchronized()
    @functools.wraps(func)
    def f():
        return impl()

    return f


def fs_cached_result(filename, force_update=False, verbose=False):
    import jacinle.io as io

    def wrapper(func):
        @synchronized()
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            if not force_update and osp.exists(filename):
                if verbose:
                    logger.info('Using cached results from "{}".'.format(filename))
                cached_value = io.load(filename)
                if cached_value is not None:
                    return cached_value

            computed_value = func(*args, **kwargs)
            if verbose:
                logger.info('Writing result cache to "{}".'.format(filename))
            io.dump(filename, computed_value)
            return computed_value
        return wrapped_func
    return wrapper

