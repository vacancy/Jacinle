# -*- coding: utf-8 -*-
# File   : thirdparty.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/23/17
#
# This file is part of Jacinle.

from tqdm import tqdm as _tqdm

__all__ = ['get_tqdm_defaults']

__tqdm_defaults = {'dynamic_ncols': True, 'ascii': True}


def get_tqdm_defaults():
    return __tqdm_defaults


def tqdm(iterable, **kwargs):
    """Wrapped tqdm, where default kwargs will be load, and support `for i in tqdm(10)` usage."""
    for k, v in get_tqdm_defaults().items():
        kwargs.setdefault(k, v)

    if type(iterable) is int:
        iterable, total = range(iterable), iterable
    else:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    if 'total' not in kwargs and total is not None:
        kwargs['total'] = total

    return _tqdm(iterable, **kwargs)


def tqdm_pbar(**kwargs):
    for k, v in get_tqdm_defaults().items():
        kwargs.setdefault(k, v)
    return _tqdm(**kwargs)
