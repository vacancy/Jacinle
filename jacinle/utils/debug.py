#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : debug.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/26/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import sys
import functools
import threading
import contextlib
import cProfile
import pstats

from .naming import func_name
from .printing import indent_text

__all__ = ['hook_exception_ipdb', 'unhook_exception_ipdb', 'exception_hook', 'decorate_exception_hook', 'timeout_ipdb', 'log_function', 'profile', 'time']


def _custom_exception_hook(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, ipdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        # ...then start the debugger in post-mortem mode.
        ipdb.post_mortem(tb)


def hook_exception_ipdb():
    if not hasattr(_custom_exception_hook, 'origin_hook'):
        _custom_exception_hook.origin_hook = sys.excepthook
        sys.excepthook = _custom_exception_hook


def unhook_exception_ipdb():
    assert hasattr(_custom_exception_hook, 'origin_hook')
    sys.excepthook = _custom_exception_hook.origin_hook


@contextlib.contextmanager
def exception_hook(enable=True):
    if enable:
        hook_exception_ipdb()
        yield
        unhook_exception_ipdb()
    else:
        yield


def decorate_exception_hook(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with exception_hook():
            return func(*args, **kwargs)
    return wrapped


def _TimeoutEnterIpdbThread(locals_, cv, timeout):
    with cv:
        if not cv.wait(timeout):
            import ipdb; ipdb.set_trace()


@contextlib.contextmanager
def timeout_ipdb(locals_, timeout=3):
    cv = threading.Condition()
    thread = threading.Thread(target=_TimeoutEnterIpdbThread, args=(locals_, cv, timeout))
    thread.start()
    yield
    with cv:
        cv.notify_all()


def log_function(function):
    print_self = False
    if '.' in function.__qualname__:
        print_self = True

    def wrapped(*args, **kwargs):
        self_info = ''
        if print_self:
            self_info = '(self={})'.format(args[0])
        # print(indent_text(f'Entering: {func_name(function)}', log_function.indent_level, indent_format='| '))
        print(indent_text(f'Entering: {func_name(function)}{self_info}', log_function.indent_level, indent_format='| '))
        arguments = ', '.join([str(arg) for arg in args])
        print(indent_text(f'Args: {arguments}', log_function.indent_level, indent_format='| '))
        print(indent_text(f'kwargs: {kwargs}', log_function.indent_level, indent_format='| '))
        log_function.indent_level += 1
        rv = 'exception'
        try:
            rv = function(*args, **kwargs)
            return rv
        except Exception as e:
            rv = str(e)
            raise
        finally:
            log_function.indent_level -= 1
            # print(indent_text(f'Exiting: {func_name(function)}', log_function.indent_level, indent_format='| '))
            print(indent_text(f'Exiting: {func_name(function)}{self_info}', log_function.indent_level, indent_format='| '))
            print(indent_text(f'Returns: {rv}', log_function.indent_level, indent_format='| '))
    return wrapped


log_function.indent_level = 0


def _inside_log(string):
    print(indent_text(str(string), log_function.indent_level, indent_format='| '))


def _inside_print(*args, sep=' ', end='\n'):
    string = sep.join([str(arg) for arg in args])
    print(indent_text(str(string), log_function.indent_level, indent_format='| ').rstrip() + end, end='')


log_function.log = _inside_log
log_function.print= _inside_print


@contextlib.contextmanager
def profile(field='tottime', top_k=10):
    FIELDS = ['tottime', 'cumtime', None]
    assert field in FIELDS
    profiler = cProfile.Profile()
    profiler.enable()
    yield
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(field)
    stats.print_stats(top_k)


@contextlib.contextmanager
def time(name=None):
    from time import time as time_func
    if name is None:
        name = 'DEFAULT'
    print(f'[Timer::{name}] Start...')
    start = time_func()
    yield
    print(f'[Timer::{name}] End. Time elapsed = {time_func() - start}')
