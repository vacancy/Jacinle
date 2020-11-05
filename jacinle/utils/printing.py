#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : printing.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import io
import sys
import numpy as np
import collections
import contextlib

import threading
from .registry import LockRegistry

__all__ = [
    'indent_text',
    'stprint', 'stformat', 'kvprint', 'kvformat',
    'PrintToStringContext', 'print_to_string', 'print_to', 'print2format'
]


_DEFAULT_FLOAT_FORMAT = '{:.6f}'


def indent_text(text, level=1, indent_format=None, tabsize=None):
    """
    Indent a string.

    Args:
        text: (str): write your description
        level: (str): write your description
        indent_format: (str): write your description
        tabsize: (int): write your description
    """
    if indent_format is not None:
        assert tabsize is None, 'Cannot provide both indent format and tabsize.'
    if tabsize is not None:
        assert indent_format is None, 'Cannot provide both indent format and tabsize.'
        indent_format = ' ' * tabsize
    if indent_format is None and tabsize is None:
        indent_format = '  '
    indent_format = indent_format * level
    return indent_format + text.replace('\n', '\n' + indent_format)


def format_printable_data(data, float_format=_DEFAULT_FLOAT_FORMAT):
    """
    Format a formatted string to printable data.

    Args:
        data: (array): write your description
        float_format: (str): write your description
        _DEFAULT_FLOAT_FORMAT: (str): write your description
    """
    t = type(data)
    if t is np.ndarray:
        return 'ndarray{}, dtype={}'.format(data.shape, data.dtype)
    # Handle torch.tensor
    if 'Tensor' in str(t):
        return 'tensor{}, dtype={}'.format(tuple(data.shape), data.dtype)
    elif t is float:
        return float_format.format(data)
    else:
        return str(data)


def stprint(data, key=None, indent=0, file=None, indent_format='  ', end_format='\n', float_format=_DEFAULT_FLOAT_FORMAT, need_lock=True, max_depth=100):
    """
    Structure print.

    Example:

        >>> data = dict(a=np.zeros(shape=(10, 10)), b=3)
        >>> stprint(data)
        dict{
            a: ndarray(10, 10), dtype=float64
            b: 3
        }

    Args:
        data: data to be print. Currently support Sequnce, Mappings and primitive types.
        key: for recursion calls. Do not use it if you don't know how it works.
        indent: indent level.
    """

    if file is None:
        file = sys.stdout

    def _indent_print(msg, indent, prefix=None):
        """
        Prints a message to the given message.

        Args:
            msg: (str): write your description
            indent: (int): write your description
            prefix: (str): write your description
        """
        print(indent_format * indent, end='', file=file)
        if prefix is not None:
            print(prefix, end='', file=file)
        print(msg, end=end_format, file=file)

    def _inner(data, indent, key, max_depth):
        """
        Prints a nested dict.

        Args:
            data: (dict): write your description
            indent: (todo): write your description
            key: (str): write your description
            max_depth: (int): write your description
        """
        t = type(data)
        if t is tuple:
            if max_depth == 0:
                _indent_print('(tuple of length {}) ...'.format(len(data)), indent, prefix=key)
                return
            _indent_print('tuple[', indent, prefix=key)
            for v in data:
                _inner(v, indent=indent + 1, key=None, max_depth=max_depth - 1)
            _indent_print(']', indent)
        elif t is list:
            if max_depth == 0:
                _indent_print('(list of length {}) ...'.format(len(data)), indent, prefix=key)
                return
            _indent_print('list[', indent, prefix=key)
            for v in data:
                _inner(v, indent=indent + 1, key=None, max_depth=max_depth - 1)
            _indent_print(']', indent)
        elif t in (dict, collections.OrderedDict):
            if max_depth == 0:
                _indent_print('(dict of length {}) ...'.format(len(data)), indent, prefix=key)
                return
            typename = 'dict' if t is dict else 'ordered_dict'
            keys = sorted(data.keys()) if t is dict else data.keys()
            _indent_print(typename + '{', indent, prefix=key)
            for k in keys:
                v = data[k]
                _inner(v, indent=indent + 1, key='{}: '.format(k), max_depth=max_depth - 1)
            _indent_print('}', indent)
        else:
            _indent_print(format_printable_data(data, float_format=float_format), indent, prefix=key)

    with stprint.locks.synchronized(file, need_lock):
        _inner(data, indent=indent, key=key, max_depth=max_depth)

    del _inner


stprint.locks = LockRegistry()


def stformat(data, key=None, indent=0, max_depth=100, **kwargs):
    """
    Stformat data to stdout.

    Args:
        data: (array): write your description
        key: (str): write your description
        indent: (str): write your description
        max_depth: (int): write your description
    """
    return print2format(stprint)(data, key=key, indent=indent, need_lock=False, max_depth=max_depth, **kwargs)


def kvprint(data, indent=0, sep=' : ', end='\n', max_key_len=None, file=None, float_format=_DEFAULT_FLOAT_FORMAT, need_lock=True):
    """
    Print the kvprint data.

    Args:
        data: (dict): write your description
        indent: (int): write your description
        sep: (todo): write your description
        end: (int): write your description
        max_key_len: (int): write your description
        file: (str): write your description
        float_format: (str): write your description
        _DEFAULT_FLOAT_FORMAT: (str): write your description
        need_lock: (todo): write your description
    """
    if len(data) == 0:
        return
    with kvprint.locks.synchronized(file, need_lock):
        keys = sorted(data.keys())
        lens = list(map(len, keys))
        if max_key_len is not None:
            max_len = max_key_len
        else:
            max_len = max(lens)
        for k in keys:
            print('  ' * indent, end='')
            print(k + ' ' * max(max_len - len(k), 0), format_printable_data(data[k], float_format=float_format), sep=sep, end=end, file=file, flush=True)


kvprint.locks = LockRegistry()


def kvformat(data, indent=0, sep=' : ', end='\n', max_key_len=None):
    """
    Kvformat data

    Args:
        data: (array): write your description
        indent: (int): write your description
        sep: (todo): write your description
        end: (int): write your description
        max_key_len: (int): write your description
    """
    return print2format(kvprint)(data, indent=indent, sep=sep, end=end, max_key_len=max_key_len, need_lock=False)


class PrintToStringContext(object):
    __global_locks = LockRegistry()

    def __init__(self, target='STDOUT', stream=None, need_lock=True):
        """
        Initialize the stream.

        Args:
            self: (todo): write your description
            target: (todo): write your description
            stream: (todo): write your description
            need_lock: (todo): write your description
        """
        assert target in ('STDOUT', 'STDERR')
        self._target = target
        self._need_lock = need_lock
        if stream is None:
            self._stream = io.StringIO()
        else:
            self._stream = stream
        self._stream_lock = threading.Lock()
        self._backup = None
        self._value = None

    def _swap(self, rhs):
        """
        Swap the right - op.

        Args:
            self: (todo): write your description
            rhs: (todo): write your description
        """
        if self._target == 'STDOUT':
            sys.stdout, rhs = rhs, sys.stdout
        else:
            sys.stderr, rhs = rhs, sys.stderr

        return rhs

    def __enter__(self):
        """
        Starts the stream.

        Args:
            self: (todo): write your description
        """
        if self._need_lock:
            self.__global_locks[self._target].acquire()
        self._backup = self._swap(self._stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when the given exception is raised.

        Args:
            self: (todo): write your description
            exc_type: (todo): write your description
            exc_val: (todo): write your description
            exc_tb: (todo): write your description
        """
        self._stream = self._swap(self._backup)
        if self._need_lock:
            self.__global_locks[self._target].release()

    def _ensure_value(self):
        """
        Close the stream.

        Args:
            self: (todo): write your description
        """
        if self._value is None:
            self._value = self._stream.getvalue()
            self._stream.close()

    def get(self):
        """
        Get the value of this field.

        Args:
            self: (todo): write your description
        """
        self._ensure_value()
        return self._value


def print_to_string(target='STDOUT'):
    """
    Prints the string to a string

    Args:
        target: (str): write your description
    """
    return PrintToStringContext(target, need_lock=True)


@contextlib.contextmanager
def print_to(print_func, target='STDOUT', rstrip=True):
    """
    Prints a string of the given function.

    Args:
        print_func: (todo): write your description
        target: (str): write your description
        rstrip: (str): write your description
    """
    with PrintToStringContext(target, need_lock=True) as ctx:
        yield
    out_str = ctx.get()
    if rstrip:
        out_str = out_str.rstrip()
    print_func(out_str)


def print2format(print_func):
    """
    Print a formatted string to print_func.

    Args:
        print_func: (todo): write your description
    """
    def format_func(*args, **kwargs):
        """
        Formats a function that formats.

        Args:
        """
        f = io.StringIO()
        print_func(*args, file=f, **kwargs)
        value = f.getvalue()
        f.close()
        return value
    return format_func
