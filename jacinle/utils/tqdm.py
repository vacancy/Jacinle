#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : tqdm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/23/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import threading

from tqdm import tqdm as _tqdm
from .meta import gofor

__all__ = ['get_tqdm_defaults', 'get_current_tqdm', 'tqdm', 'tqdm_pbar', 'tqdm_gofor', 'tqdm_zip']

__tqdm_defaults = {'dynamic_ncols': True, 'ascii': True}


def get_tqdm_defaults():
    """
    Return the default value for the default tqdm.

    Args:
    """
    return __tqdm_defaults


def get_current_tqdm():
    """
    Return current tqdm stack.

    Args:
    """
    _init_tqdm_stack()
    assert len(get_current_tqdm._stack.data) > 0, 'No registered tqdm.'
    return get_current_tqdm._stack.data[0]

get_current_tqdm._stack = threading.local()
get_current_tqdm._stack.data = list()

def _init_tqdm_stack():
    """
    Initialize tqdm stack.

    Args:
    """
    if not hasattr(get_current_tqdm._stack, 'data'):
        get_current_tqdm._stack.data = list()


def tqdm(iterable, **kwargs):
    """Wrapped tqdm, where default kwargs will be load, and support `for i in tqdm(10)` usage."""
    for k, v in get_tqdm_defaults().items():
        kwargs.setdefault(k, v)

    if type(iterable) is int:
        iterable, total = range(iterable), iterable
    elif type(iterable) is float:
        iterable, total = range(int(iterable)), iterable
    else:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    if 'total' not in kwargs and total is not None:
        kwargs['total'] = total

    with _tqdm(**kwargs) as pbar:
        _init_tqdm_stack()
        get_current_tqdm._stack.data.append(pbar)
        try:
            for data in iterable:
                yield data
                pbar.update()
        finally:
            get_current_tqdm._stack.data.pop()


def tqdm_pbar(**kwargs):
    """
    Create a tqdm bar.

    Args:
    """
    for k, v in get_tqdm_defaults().items():
        kwargs.setdefault(k, v)
    return _tqdm(**kwargs)


def tqdm_gofor(iterable, **kwargs):
    """
    Go through a tqdm tqdm.

    Args:
        iterable: (todo): write your description
    """
    try:
        total = len(iterable)
    except TypeError:
        total = None
    kwargs.setdefault('total', total)
    return tqdm(gofor(iterable), **kwargs)


def tqdm_zip(*iterable, **kwargs):
    """
    Zip a tqdm tqdm iterable sequences.

    Args:
        iterable: (todo): write your description
    """
    try:
        total = len(iterable[0])
    except TypeError:
        total = None

    kwargs.setdefault('total', total)
    return tqdm(zip(*iterable), **kwargs)

