# -*- coding: utf-8 -*-
# File   : cache.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 19/01/2018
#
# This file is part of Jacinle.

import functools
import threading
import collections

from .meta import synchronized

__all__ = ['cached_property', 'cached_result']


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
        with self.__mutex[instance]:
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


# TODO::
# def fs_cached_result(cache_key, force_update=False):
#     def wrapper(func):
#         @synchronized()
#         @functools.wraps(func)
#         def wrapped_func(*args, **kwargs):
#             if get_env('dir.cache') is None:
#                 io.make_env_dir('dir.cache', osp.join(get_env('dir.root'), 'cache'))
#
#             nonlocal cache_key
#             cache_key = io.assert_extension(cache_key, '.cache.pkl')
#             cache_file = osp.join(get_env('dir.cache'), cache_key)
#
#             if not force_update:
#                 cached_value = io.load(cache_file)
#                 if cached_value is not None:
#                     return cached_value
#
#             computed_value = func(*args, **kwargs)
#             io.dump(cache_file, computed_value)
#             return computed_value
#         return wrapped_func
#     return wrapper
