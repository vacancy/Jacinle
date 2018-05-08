#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : printing.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 18/01/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import io
import sys
import numpy as np
import collections

from .registry import LockRegistry

__all__ = ['stprint', 'stformat', 'kvprint', 'kvformat', 'print_to_string', 'print2format']


def _indent_print(msg, indent, prefix=None, end='\n', file=None):
    print(*['  '] * indent, end='', file=file)
    if prefix is not None:
        print(prefix, end='', file=file)
    print(msg, end=end, file=file)


def stprint(data, key=None, indent=0, file=None, need_lock=True):
    """
    Structure print. Usage:

    ```
    data = dict(a=np.zeros(shape=(10, 10)), b=3)
    stprint(data)
    ```

    and you will get:

    ```
    dict{
        a: ndarray(10, 10), dtype=float64
        b: 3
    }
    ```

    :param data: Data you want to print.
    :param key: Output prefix, internal usage only.
    :param indent: Indent level of the print, internal usage only.
    """
    t = type(data)
    if file is None:
        file = sys.stdout

    with stprint.locks.synchronized(file, need_lock):
        if t is tuple:
            _indent_print('tuple[', indent, prefix=key, file=file)
            for v in data:
                stprint(v, indent=indent + 1, file=file, need_lock=False)
            _indent_print(']', indent, file=file)
        elif t is list:
            _indent_print('list[', indent, prefix=key, file=file)
            for v in data:
                stprint(v, indent=indent + 1, file=file, need_lock=False)
            _indent_print(']', indent, file=file)
        elif t in (dict, collections.OrderedDict):
            typename = 'dict' if t is dict else 'ordered_dict'
            keys = sorted(data.keys()) if t is dict else data.keys()
            _indent_print(typename, indent, prefix=key, file=file)
            for k in keys:
                v = data[k]
                stprint(v, indent=indent + 1, key='{}: '.format(k), file=file, need_lock=False)
            _indent_print('}', indent, file=file)
        elif t is np.ndarray:
            _indent_print('ndarray{}, dtype={}'.format(data.shape, data.dtype), indent, prefix=key, file=file)
        else:
            _indent_print(data, indent, prefix=key, file=file)


stprint.locks = LockRegistry()


def stformat(data, key=None, indent=0):
    return print2format(stprint)(data, key=key, indent=indent, need_lock=False)


def kvprint(data, sep=': ', end='\n', file=None, need_lock=True):
    with kvprint.locks.synchronized(file, need_lock):
        keys = sorted(data.keys())
        lens = list(map(len, keys))
        max_len = max(lens)
        for k in keys:
            print(k + ' ' * (max_len - len(k)), data[k], sep=sep, end=end, file=file, flush=True)


kvprint.locks = LockRegistry()


def kvformat(data, sep=':', end='\n'):
    return print2format(kvprint)(data, sep=sep, end=end, need_lock=False)


class _PrintToStringContext(object):
    __global_locks = LockRegistry()

    def __init__(self, target='STDOUT', need_lock=True):
        assert target in ('STDOUT', 'STDERR')
        self._target = target
        self._need_lock = need_lock
        self._stream = io.StringIO()
        self._backup = None
        self._value = None

    def _swap(self, rhs):
        if self._target == 'STDOUT':
            sys.stdout, rhs = rhs, sys.stdout
        else:
            sys.stderr, rhs = rhs, sys.stderr

        return rhs

    def __enter__(self):
        if self._need_lock:
            self.__global_locks[self._target].acquire()
        self._backup = self._swap(self._stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stream = self._swap(self._backup)
        if self._need_lock:
            self.__global_locks[self._target].release()

    def _ensure_value(self):
        if self._value is None:
            self._value = self._stream.getvalue()
            self._stream.close()

    def get(self):
        self._ensure_value()
        return self._value


def print_to_string(target='STDOUT'):
    return _PrintToStringContext(target, need_lock=True)


def print2format(print_func):
    def format_func(*args, **kwargs):
        f = io.StringIO()
        print_func(*args, file=f, **kwargs)
        value = f.getvalue()
        f.close()
        return value
    return format_func
