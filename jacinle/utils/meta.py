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
import time
import collections
import collections.abc
import threading
import contextlib

from typing import Any, Optional, Union, Iterable, Tuple, List, Dict, Callable

__all__ = [
    'UNSET',
    'gofor',
    'run_once', 'try_run',
    'map_exec', 'filter_exec', 'first', 'first_n', 'stmap',
    'method2func', 'map_exec_method',
    'decorator_with_optional_args',
    'cond_with', 'cond_with_group',
    'merge_iterable',
    'dict_deep_update', 'dict_deep_kv', 'dict_deep_keys',
    'assert_instance', 'assert_none', 'assert_notnone',
    'notnone_property',
    'synchronized',
    'timeout', 'Clock',
    'make_dummy_func',
    'repr_from_str'
]

UNSET = object()
"""A special object to indicate that a value is not set."""


def gofor(v: Iterable[Any]) -> Iterable[Tuple[Any, Any]]:
    """A go-style for loop for dict, list, tuple, set, etc.

    - dict: for key, value in gofor(dict):
    - list, tuple, set: for index, value in gofor(list):

    Args:
        v: the iterable object.
    """
    if isinstance(v, collections.abc.Mapping):
        return v.items()
    assert_instance(v, collections.abc.Iterable)
    return enumerate(v)


def run_once(func):
    """A decorator to run a function only once."""
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
    """A function that tries to run a function, and returns None if it fails (without raising exceptions)."""
    try:
        return lambda_()
    except Exception:
        return None


def map_exec(func, *iterables: Iterable[Any]) -> List[Any]:
    """Execute a function on each element of the iterables, and return the results."""
    return list(map(func, *iterables))


def filter_exec(func, iterable: Iterable[Any]) -> List[Any]:
    """Execute a filter function on each element of the iterable, and return the results."""
    return list(filter(func, iterable))


def first(iterable: Iterable[Any], default: Any = None) -> Any:
    """Get the first element of an iterable. If the iterable is empty, return the default value.

    Args:
        iterable: the iterable object.
        default: the default value.

    Returns:
        the first element of the iterable, or the default value.
    """
    try:
        return next(iter(iterable))
    except StopIteration:
        return default


def first_n(iterable: Iterable, n: int = 10) -> Optional[List[Any]]:
    """Get the first n elements of an iterable. If the iterable has less than n elements, return None.

    Args:
        iterable: the iterable object.
        n: the number of elements to get.

    Returns:
        the first n elements of the iterable, or None.
    """
    def gen():
        it = iter(iterable)
        for i in range(n):
            try:
                yield next(it)
            except StopIteration:
                return

    return list(gen())


def stmap(func, iterable: Iterable[Any]) -> Iterable[Any]:
    """A map function that recursively follows the structure of the iterable.

    - list, tuple: return ``[func(v) for v in iterable]``
    - set: return ``{func(v) for v in iterable}``
    - dict: return ``{k: func(v) for k, v in iterable.items()}``

    Args:
        func: the function to be applied.
        iterable: the iterable object.

    Returns:
        the mapped iterable.
    """
    if isinstance(iterable, str):
        return func(iterable)
    elif isinstance(iterable, collections.abc.Sequence):
        return [stmap(func, v) for v in iterable]
    elif isinstance(iterable, collections.abc.Set):
        return {stmap(func, v) for v in iterable}
    elif isinstance(iterable, (dict, collections.abc.Mapping)):
        return {k: stmap(func, v) for k, v in iterable.items()}
    else:
        return func(iterable)


def method2func(method_name: str) -> Callable:
    """Convert a method name to a function that calls the method. Equivalent to ``lambda x: x.method_name()``.

    Args:
        method_name: the method name.
    """
    return lambda x: getattr(x, method_name)()


def map_exec_method(method_name, iterable: Iterable[Any]) -> List[Any]:
    """Execute a method on each element of the iterable, and return the results.

    Args:
        method_name: the method name.
        iterable: the iterable object.
    """
    return list(map(method2func(method_name), iterable))


def decorator_with_optional_args(func=None, *, is_method=False):
    """Make a decorator that can be used with or without arguments.

    Args:
        func: the function to be decorated.
        is_method: whether the function is a method.

    Example:
        .. code-block:: python

            @decorator_with_optional_args
            def my_decorator(func=None, *, a=1, b=2):
                def wrapper(func):
                    @functools.wraps(func)
                    def new_func(*args, **kwargs):
                        print(f'Calling {func.__name__} with a={a}, b={b}')
                        return func(*args, **kwargs)
                    return new_func
                return wrapper

            @my_decorator
            def func1():
                pass  # Calling func1 with a=1, b=2

            @my_decorator(a=2)
            def func2():
           pass  # Calling func2 with a=2, b=2

    """
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
def cond_with(with_statement, cond: bool):
    """A context manager that runs a with statement only if the condition is true."""
    if cond:
        with with_statement as res:
            yield res
    else:
        yield


@contextlib.contextmanager
def cond_with_group(cond: bool, *with_statement):
    """A context manager that runs a group of with statements only if the condition is true."""
    if cond:
        with contextlib.ExitStack() as stack:
            res = [stack.enter_context(ctx) for ctx in with_statement]
            yield res
    else:
        yield


def merge_iterable(v1, v2):
    """Merge two iterables into a single iterable.

    - list, tuple: return ``v1 + v2``
    - set: return ``v1 | v2``
    - dict: return ``{**v1, **v2}``

    Args:
        v1: the first iterable.
        v2: the second iterable.

    Returns:
        the merged iterable.
    """

    assert issubclass(type(v1), type(v2)) or issubclass(type(v2), type(v1))
    if isinstance(v1, (dict, set)):
        v = v1.copy().update(v2)
        return v

    return v1 + v2


def dict_deep_update(a: Dict[Any, Any], b: Dict[Any, Any]):
    """Update a dictionary recursively.

    Args:
        a: the dictionary to be updated.
        b: the dictionary to update from.
    """
    for key in b:
        if key in a and type(b[key]) is dict:
            dict_deep_update(a[key], b[key])
        else:
            a[key] = b[key]


def dict_deep_kv(d: Dict[Any, Any], sort: bool = True, sep='.', allow_dict: bool = False) -> Dict[str, Any]:
    """Get a flattened dictionary with keys as the path to the value.

    Example:
        >>> d = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> dict_deep_kv(d)
        {'a': 1, 'b.c': 2, 'b.d': 3}

    Args:
        d: the dictionary.
        sort: whether to sort the keys.
        sep: the separator between keys.
        allow_dict: whether to allow dictionary values.
    """

    # Not using collections.Sequence to avoid infinite recursion.
    assert isinstance(d, (tuple, list, dict))
    result = list()

    def _dfs(current, prefix=None):
        for key, value in gofor(current):
            current_key = key if prefix is None else prefix + sep + str(key)
            if isinstance(current[key], (tuple, list, dict)):
                if allow_dict:
                    result.append((current_key, value))
                _dfs(current[key], current_key)
            else:
                result.append((current_key, value))

    _dfs(d)
    if sort:
        result.sort(key=operator.itemgetter(0))
    return result


def dict_deep_keys(d: Dict[Any, Any], sort: bool = True, sep='.', allow_dict: bool = False) -> List[str]:
    """Get the keys of a flattened dictionary.

    Args:
        d: the dictionary.
        sort: whether to sort the keys.
        sep: the separator between keys.
        allow_dict: whether to allow dictionary values.

    Returns:
        a list of keys.

    See also:

        :func:`dict_deep_kv`
    """
    kv = dict_deep_kv(d, sort=sort, sep=sep, allow_dict=allow_dict)
    return [i[0] for i in kv]


def assert_instance(ins: Any, clz: Union[type, Tuple[type, ...]], msg: str = None):
    """Assert that an instance is of a certain type."""
    msg = msg or '{} (of type{}) is not of type {}'.format(ins, type(ins), clz)
    assert isinstance(ins, clz), msg


def assert_none(ins: Any, msg: str = None):
    """Assert that the input is None."""
    msg = msg or '{} is not None'.format(ins)
    assert ins is None, msg


def assert_notnone(ins: Any, msg: str = None, name: str = None):
    """Assert that the input is not None.

    Args:
        ins: the input.
        msg: the error message. If not specified, a default message ``{name} is None`` will be used.
        name: the name of the input.
    """
    msg = msg or '{} is None'.format(name)
    assert ins is not None, msg


def notnone_property(fget):
    """A property that raises an error if the value is None."""

    @functools.wraps(fget)
    def wrapped(self):
        v = fget(self)
        assert v is not None, '{}.{} can not be None, maybe not set yet'.format(
                type(self).__name__, fget.__name__)
        return v

    return property(wrapped)


@decorator_with_optional_args
def synchronized(mutex=None):
    """A decorator that synchronizes the execution of a function.

    Args:
        mutex: the mutex to use. If not specified, a new threading mutex will be created.
    """
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


def timeout(timeout: float, fps: Optional[float] = None):
    """A decorator that raises a TimeoutError if the execution time of the function exceeds the timeout.

    Args:
        timeout (float): timeout in seconds.
        fps (float): an optional fps to control the maximum number of iterations.

    Example:
        .. code-block:: python

            import time
            from jacinle.utils.meta import timeout
            for _ in timeout(5.1):
                print('hello')
                time.sleep(1)
    """
    t0 = time.time()
    if fps is not None:
        max_iterations = int(timeout * fps)
    iterations = 0
    while time.time() - t0 < timeout:
        iterations += 1
        if fps is not None and iterations > max_iterations:
            break
        yield


class Clock(object):
    """A clock that can be used to measure the time."""

    def __init__(self, tick: Optional[float] = None):
        """Initialize the clock.

        Args:
            tick: the time (second) for each tick of the clock.
        """
        self.last_time = time.time()
        self.timeout = tick

    def tick(self, timeout=None):
        """Tick the clock.

        Args:
            timeout: the timeout in seconds. If not specified, the timeout in the constructor will be used.
        """
        if timeout is None:
            timeout = self.timeout
            if timeout is None:
                raise ValueError('timeout is None')
        t = time.time()
        if t - self.last_time >= timeout:
            self.last_time = t
            return False
        time.sleep(timeout - (t - self.last_time))
        self.last_time = time.time()
        return True


def make_dummy_func(message=None):
    """Make a dummy function that raises an error when called."""
    def func(*args, **kwargs):
        raise NotImplementedError(message)
    return func


def repr_from_str(self):
    """A helper function to generate the repr string from the __str__ method.

    Example:
        .. code-block:: python

            class Foo(object):
                def __str__(self):
                    return 'Foo'

                __repr__ = repr_from_str

            print(Foo())  # Foo
            print(repr(Foo()))  # Foo[Foo]
    """
    return '{}<{}>'.format(self.__class__.__name__, self.__str__())
