#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : meta.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import functools
import operator
import six
import collections
import threading
import contextlib

__all__ = [
    'gofor',
    'run_once', 'try_run',
    'map_exec', 'filter_exec', 'first_n', 'stmap',
    'method2func', 'map_exec_method',
    'decorator_with_optional_args',
    'cond_with', 'cond_with_group',
    'merge_iterable',
    'dict_deep_update', 'dict_deep_kv', 'dict_deep_keys',
    'assert_instance', 'assert_none', 'assert_notnone',
    'notnone_property',
    'synchronized',
    'make_dummy_func'
]


def gofor(v):
    if isinstance(v, collections.Mapping):
        return v.items()
    assert_instance(v, collections.Iterable)
    return enumerate(v)


def run_once(func):
    has_run = False

    @synchronized
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        nonlocal has_run
        if not has_run:
            has_run = True
            return func(*args, **kwargs)
        else:
            return
    return new_func


def try_run(lambda_):
    try:
        return lambda_()
    except Exception:
        return None


def map_exec(func, *iterables):
    return list(map(func, *iterables))


def filter_exec(func, iterable):
    return list(filter(func, iterable))


def first_n(iterable, n=10):
    def gen():
        it = iter(iterable)
        for i in range(n):
            try:
                yield next(it)
            except StopIteration:
                return

    return list(gen())


def stmap(func, iterable):
    if isinstance(iterable, six.string_types):
        return func(iterable)
    elif isinstance(iterable, (collections.Sequence, collections.UserList)):
        return [stmap(func, v) for v in iterable]
    elif isinstance(iterable, collections.Set):
        return {stmap(func, v) for v in iterable}
    elif isinstance(iterable, (collections.Mapping, collections.UserDict)):
        return {k: stmap(func, v) for k, v in iterable.items()}
    else:
        return func(iterable)


def method2func(method_name):
    return lambda x: getattr(x, method_name)()


def map_exec_method(method_name, iterable):
    return list(map(method2func(method_name), iterable))


def decorator_with_optional_args(func=None, *, is_method=False):
    def wrapper(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            if is_method:
                if len(args) == 1:
                    return f(*args, **kwargs)
                elif len(args) == 2:
                    return f(args[0], **kwargs)(args[1])
                else:
                    raise ValueError('Decorator supports 0 or 1 positional arguments as the function to be wrapped.')
            else:
                if len(args) == 0:
                    return f(**kwargs)
                elif len(args) == 1:
                    return f(**kwargs)(args[0])
                else:
                    raise ValueError('Decorator supports 0 or 1 positional arguments as the function to be wrapped.')
        return wrapped

    if func is not None:
        return wrapper(func)
    else:
        return wrapper


@contextlib.contextmanager
def cond_with(with_statement, cond):
    if cond:
        with with_statement as res:
            yield res
    else:
        yield


@contextlib.contextmanager
def cond_with_group(cond, *with_statement):
    if cond:
        with contextlib.ExitStack() as stack:
            res = [stack.enter_context(ctx) for ctx in with_statement]
            yield res
    else:
        yield


def merge_iterable(v1, v2):
    assert issubclass(type(v1), type(v2)) or issubclass(type(v2), type(v1))
    if isinstance(v1, (dict, set)):
        v = v1.copy().update(v2)
        return v

    return v1 + v2


def dict_deep_update(a, b):
    for key in b:
        if key in a and type(b[key]) is dict:
            dict_deep_update(a[key], b[key])
        else:
            a[key] = b[key]


def dict_deep_kv(d, sort=True, sep='.', allow_dict=False):
    # Not using collections.Sequence to avoid infinite recursion.
    assert isinstance(d, (tuple, list, collections.Mapping))
    result = list()

    def _dfs(current, prefix=None):
        for key, value in gofor(current):
            current_key = key if prefix is None else prefix + sep + str(key)
            if isinstance(current[key], (tuple, list, collections.Mapping)):
                if allow_dict:
                    result.append((current_key, value))
                _dfs(current[key], current_key)
            else:
                result.append((current_key, value))

    _dfs(d)
    if sort:
        result.sort(key=operator.itemgetter(0))
    return result


def dict_deep_keys(d, sort=True, sep='.', allow_dict=True):
    kv = dict_deep_kv(d, sort=sort, sep=sep, allow_dict=allow_dict)
    return [i[0] for i in kv]


def assert_instance(ins, clz, msg=None):
    msg = msg or '{} (of type{}) is not of type {}'.format(ins, type(ins), clz)
    assert isinstance(ins, clz), msg


def assert_none(ins, msg=None):
    msg = msg or '{} is not None'.format(ins)
    assert ins is None, msg


def assert_notnone(ins, msg=None, name='instance'):
    msg = msg or '{} is None'.format(name)
    assert ins is not None, msg


class notnone_property:
    def __init__(self, fget):
        self.fget = fget
        self.__module__ = fget.__module__
        self.__name__ = fget.__name__
        self.__doc__ = fget.__doc__
        self.__prop_key  = '{}_{}'.format(
            fget.__name__, id(fget))

    def __get__(self, instance, owner):
        if instance is None:
            return self.fget
        v = self.fget(instance)
        assert v is not None, '{}.{} can not be None, maybe not set yet'.format(
                type(instance).__name__, self.__name__)
        return v


@decorator_with_optional_args
def synchronized(mutex=None):
    if mutex is None:
        mutex = threading.Lock()

    def wrapper(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            with mutex:
                return func(*args, **kwargs)
        wrapped_func.__sync_mutex__ = mutex
        return wrapped_func

    return wrapper


def make_dummy_func(message=None):
    def func(*args, **kwargs):
        raise NotImplementedError(message)
    return func

