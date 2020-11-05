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
    """
    Go through a sequence of dicts or dicts.

    Args:
        v: (dict): write your description
    """
    if isinstance(v, collections.Mapping):
        return v.items()
    assert_instance(v, collections.Iterable)
    return enumerate(v)


def run_once(func):
    """
    Decorator to run a function.

    Args:
        func: (todo): write your description
    """
    has_run = False

    @synchronized
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        """
        Decorator to create a new function.

        Args:
        """
        nonlocal has_run
        if not has_run:
            has_run = True
            return func(*args, **kwargs)
        else:
            return
    return new_func


def try_run(lambda_):
    """
    Try to find function.

    Args:
        lambda_: (float): write your description
    """
    try:
        return lambda_()
    except Exception:
        return None


def map_exec(func, *iterables):
    """
    Map a function over a list.

    Args:
        func: (todo): write your description
        iterables: (todo): write your description
    """
    return list(map(func, *iterables))


def filter_exec(func, iterable):
    """
    Filter a function using the given iterable.

    Args:
        func: (todo): write your description
        iterable: (todo): write your description
    """
    return list(filter(func, iterable))


def first_n(iterable, n=10):
    """
    Returns the first n items from iterable.

    Args:
        iterable: (todo): write your description
        n: (int): write your description
    """
    def gen():
        """
        Yields the first n - length.

        Args:
        """
        it = iter(iterable)
        for i in range(n):
            try:
                yield next(it)
            except StopIteration:
                return

    return list(gen())


def stmap(func, iterable):
    """
    Stmap a function over iterable.

    Args:
        func: (todo): write your description
        iterable: (dict): write your description
    """
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
    """
    Decorator for method methods method

    Args:
        method_name: (str): write your description
    """
    return lambda x: getattr(x, method_name)()


def map_exec_method(method_name, iterable):
    """
    Map an iterable to an iterable.

    Args:
        method_name: (str): write your description
        iterable: (todo): write your description
    """
    return list(map(method2func(method_name), iterable))


def decorator_with_optional_args(func=None, *, is_method=False):
    """
    Decorator for decorate a decorator.

    Args:
        func: (todo): write your description
        is_method: (str): write your description
    """
    def wrapper(f):
        """
        Decorator to wrap the wrapped function.

        Args:
            f: (int): write your description
        """
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            """
            Decorator to wrap a method.

            Args:
            """
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
    """
    Yields the given statement if_statement.

    Args:
        with_statement: (bool): write your description
        cond: (todo): write your description
    """
    if cond:
        with with_statement as res:
            yield res
    else:
        yield


@contextlib.contextmanager
def cond_with_group(cond, *with_statement):
    """
    Context manager that yields groups of - like group.

    Args:
        cond: (todo): write your description
        with_statement: (bool): write your description
    """
    if cond:
        with contextlib.ExitStack() as stack:
            res = [stack.enter_context(ctx) for ctx in with_statement]
            yield res
    else:
        yield


def merge_iterable(v1, v2):
    """
    Merge two iterable objects.

    Args:
        v1: (array): write your description
        v2: (todo): write your description
    """
    assert issubclass(type(v1), type(v2)) or issubclass(type(v2), type(v1))
    if isinstance(v1, (dict, set)):
        v = v1.copy().update(v2)
        return v

    return v1 + v2


def dict_deep_update(a, b):
    """
    Update dict b into dict b.

    Args:
        a: (todo): write your description
        b: (todo): write your description
    """
    for key in b:
        if key in a and type(b[key]) is dict:
            dict_deep_update(a[key], b[key])
        else:
            a[key] = b[key]


def dict_deep_kv(d, sort=True, sep='.', allow_dict=False):
    """
    Convert a dictionary of dicts.

    Args:
        d: (todo): write your description
        sort: (callable): write your description
        sep: (todo): write your description
        allow_dict: (bool): write your description
    """
    # Not using collections.Sequence to avoid infinite recursion.
    assert isinstance(d, (tuple, list, collections.Mapping))
    result = list()

    def _dfs(current, prefix=None):
        """
        Go through a list of keys.

        Args:
            current: (todo): write your description
            prefix: (str): write your description
        """
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
    """
    Return a list of all keys in a dictionary.

    Args:
        d: (todo): write your description
        sort: (callable): write your description
        sep: (todo): write your description
        allow_dict: (bool): write your description
    """
    kv = dict_deep_kv(d, sort=sort, sep=sep, allow_dict=allow_dict)
    return [i[0] for i in kv]


def assert_instance(ins, clz, msg=None):
    """
    Fail if clzerror if the same length.

    Args:
        ins: (todo): write your description
        clz: (todo): write your description
        msg: (str): write your description
    """
    msg = msg or '{} (of type{}) is not of type {}'.format(ins, type(ins), clz)
    assert isinstance(ins, clz), msg


def assert_none(ins, msg=None):
    """
    Asserts that the first is none.

    Args:
        ins: (todo): write your description
        msg: (str): write your description
    """
    msg = msg or '{} is not None'.format(ins)
    assert ins is None, msg


def assert_notnone(ins, msg=None, name='instance'):
    """
    Asserts that the assertion is not none.

    Args:
        ins: (todo): write your description
        msg: (str): write your description
        name: (str): write your description
    """
    msg = msg or '{} is None'.format(name)
    assert ins is not None, msg


class notnone_property:
    def __init__(self, fget):
        """
        Initialize the object

        Args:
            self: (todo): write your description
            fget: (callable): write your description
        """
        self.fget = fget
        self.__module__ = fget.__module__
        self.__name__ = fget.__name__
        self.__doc__ = fget.__doc__
        self.__prop_key  = '{}_{}'.format(
            fget.__name__, id(fget))

    def __get__(self, instance, owner):
        """
        Return the value of the attribute

        Args:
            self: (dict): write your description
            instance: (todo): write your description
            owner: (todo): write your description
        """
        if instance is None:
            return self.fget
        v = self.fget(instance)
        assert v is not None, '{}.{} can not be None, maybe not set yet'.format(
                type(instance).__name__, self.__name__)
        return v


@decorator_with_optional_args
def synchronized(mutex=None):
    """
    Decorator for a function async function.

    Args:
        mutex: (str): write your description
    """
    if mutex is None:
        mutex = threading.Lock()

    def wrapper(func):
        """
        Decorator for the wrapped function.

        Args:
            func: (callable): write your description
        """
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            """
            Decorator for the wrapped function.

            Args:
            """
            with mutex:
                return func(*args, **kwargs)
        wrapped_func.__sync_mutex__ = mutex
        return wrapped_func

    return wrapper


def make_dummy_func(message=None):
    """
    Make a function that creates a callable.

    Args:
        message: (str): write your description
    """
    def func(*args, **kwargs):
        """
        Decor function.

        Args:
        """
        raise NotImplementedError(message)
    return func

