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
import os
import sys
import numpy as np
import collections
import contextlib
import threading
from typing import Optional, Callable

from .registry import LockRegistry

try:
    import torch
except ImportError:
    torch = None

__all__ = [
    'indent_text',
    'stprint', 'stformat', 'kvprint', 'kvformat',
    'PrintToStringContext', 'print_to_string', 'print_to', 'print2format',
    'suppress_stdout', 'suppress_stderr', 'suppress_output',
]


_DEFAULT_FLOAT_FORMAT = '{:.6f}'


def indent_text(text: str, level = 1, indent_format: Optional[str] = None, tabsize: Optional[int] = None) -> str:
    """Indent the text by the given level.

    Args:
        text: the text to be indented.
        level: the indent level.
        indent_format: the indent format. If None, use the tabsize.
        tabsize: the tab size. If None, use the default tab size (2).

    Returns:
        The indented text.
    """
    text = str(text)
    if indent_format is not None:
        assert tabsize is None, 'Cannot provide both indent format and tabsize.'
    if tabsize is not None:
        assert indent_format is None, 'Cannot provide both indent format and tabsize.'
        indent_format = ' ' * tabsize
    if indent_format is None and tabsize is None:
        indent_format = '  '
    indent_format = indent_format * level
    return indent_format + text.replace('\n', '\n' + indent_format)


def format_printable_data(data, float_format: str = _DEFAULT_FLOAT_FORMAT, indent: int = 1, indent_format: str = '  '):
    """Format the input data. It handles the following types:

    - numpy array: print the shape and dtype.
    - torch tensor: print the shape and dtype.
    - float: print with the given float format.
    - other types: use str() to print.

    Args:
        data: the data to be printed.
        float_format: the float format.
        indent: the indent level.
        indent_format: the indent format.
    """
    t = type(data)
    if t is np.ndarray:
        fmt = 'np.ndarray(shape={}, dtype={})'.format(data.shape, data.dtype)
        if data.size < 100:
            fmt += '{' + indent_text(str(data), level=indent, indent_format=indent_format).strip() + '}'
        return fmt

    if torch is not None and torch.is_tensor(data):
        fmt = 'torch.Tensor(shape={}, dtype={})'.format(tuple(data.shape), data.dtype)
        if data.numel() < 100:
            fmt += '{' + indent_text(str(data), level=indent, indent_format=indent_format).strip() + '}'
        return fmt
    elif t is float:
        return float_format.format(data)
    else:
        return str(data)


def stprint(
    data,
    key: Optional[str] = None,
    indent: int = 0,
    file: Optional[io.TextIOBase] = None,
    indent_format: str = '  ', end_format: str = '\n', float_format: str = _DEFAULT_FLOAT_FORMAT,
    need_lock: bool = True, sort_key: bool = True, max_depth: int = 100
):
    """Structure print.

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
        file: the file to print to.
        indent_format: the indent format.
        end_format: the end format.
        float_format: the float format.
        need_lock: whether to use the lock.
        sort_key: whether to sort the keys.
        max_depth: the maximum depth of the recursion.
    """
    from .container import GView

    if file is None:
        file = sys.stdout

    def _indent_print(msg, indent, prefix=None):
        print(indent_format * indent, end='', file=file)
        if prefix is not None:
            print(prefix, end='', file=file)
        print(indent_text(msg, indent, indent_format=indent_format).lstrip(), end=end_format, file=file)

    def _inner(data, indent, key, max_depth):
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
        elif t in (dict, collections.OrderedDict, collections.defaultdict, GView):
            if max_depth == 0:
                _indent_print('(dict of length {}) ...'.format(len(data)), indent, prefix=key)
                return
            typename = t.__name__
            keys = sorted(data.keys()) if t is dict and sort_key else data.keys()
            _indent_print(typename + '{', indent, prefix=key)
            for k in keys:
                v = data[k]
                _inner(v, indent=indent + 1, key='{}: '.format(k), max_depth=max_depth - 1)
            _indent_print('}', indent)
        else:
            _indent_print(format_printable_data(data, float_format=float_format, indent=indent + 1, indent_format=indent_format), indent, prefix=key)

    try:
        with stprint.locks.synchronized(file, need_lock):
            _inner(data, indent=indent, key=key, max_depth=max_depth)
    finally:
        del _inner


stprint.locks = LockRegistry()


def stformat(data, key=None, indent=0, max_depth=100, **kwargs):
    """Structure format. See :func:`stprint` for more details."""
    return print2format(stprint)(data, key=key, indent=indent, need_lock=False, max_depth=max_depth, **kwargs)


def kvprint(
    data,
    indent: int = 0, sep: str = ' : ', end: str = '\n',
    max_key_len: Optional[int] = None,
    file: Optional[io.TextIOBase] = None,
    float_format: str = _DEFAULT_FLOAT_FORMAT,
    need_lock: bool = True
):
    """Print the key-value pairs.

    Args:
        data: the data to be printed.
        indent: the indent level.
        sep: the separator between key and value.
        end: the end format.
        max_key_len: the maximum length of the key. If None, use the maximum length of the keys.
        file: the file to print to.
        float_format: the float format.
        need_lock: whether to use the lock.
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
    """Format the key-value pairs. See :func:`kvprint` for more details."""
    return print2format(kvprint)(data, indent=indent, sep=sep, end=end, max_key_len=max_key_len, need_lock=False)


class PrintToStringContext(object):
    """A context manager that redirect the print to a string.

    Example:
        >>> with PrintToStringContext() as s:
        ...     print('hello')
        >>> print(s.get())
    """
    __global_locks = LockRegistry()

    def __init__(self, target='STDOUT', stream=None, need_lock=True):
        """Initialize the context.

        Args:
            target: the target to redirect to. Can be 'STDOUT', 'STDERR'.
            stream: the stream to redirect to. If None, use a new :class:`io.StringIO`.
            need_lock: whether to use the lock.
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

    def get(self) -> str:
        """Get the string."""
        self._ensure_value()
        return self._value


def print_to_string(target='STDOUT'):
    """Create a :class:`PrintToStringContext` and return the context manager."""
    return PrintToStringContext(target, need_lock=True)


@contextlib.contextmanager
def print_to(print_func, target='STDOUT', rstrip=True):
    """Redirect the print to a function.

    Example:
        .. code-block:: python

            def print_func(s):
                print('print_func: {}'.format(s))

            with print_to(print_func):
               print('hello')

    Args:
        print_func: the function to redirect to.
        target: the target to redirect to. Can be 'STDOUT', 'STDERR'.
        rstrip: whether to remove the trailing newlines.
    """
    with PrintToStringContext(target, need_lock=True) as ctx:
        yield
    out_str = ctx.get()
    if rstrip:
        out_str = out_str.rstrip()
    print_func(out_str)


def print2format(print_func: Callable) -> Callable:
    """A helper class to convert a "print" function to a "format" function."""
    def format_func(*args, **kwargs):
        f = io.StringIO()
        print_func(*args, file=f, **kwargs)
        value = f.getvalue()
        f.close()
        return value
    return format_func


@contextlib.contextmanager
def suppress_stdout():
    """A context manager that suppress the stdout."""
    try:
        fd = sys.stdout.fileno()
    except io.UnsupportedOperation:
        yield
        return

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)

        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as CLOEXEC may be different


@contextlib.contextmanager
def suppress_stderr():
    """A context manager that suppress the stdout."""
    try:
        fd = sys.stderr.fileno()
    except io.UnsupportedOperation:
        yield
        return

    def _redirect_stdout(to):
        sys.stderr.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stderr = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)

        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as CLOEXEC may be different


@contextlib.contextmanager
def suppress_output():
    """A context manager that suppress the stdout and stderr."""
    with suppress_stdout(), suppress_stderr():
            yield