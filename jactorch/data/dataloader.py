#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dataloader.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
from torch.utils.data.dataloader import DataLoader, default_collate

from jacinle.random import reset_global_seed, gen_seed

__all__ = ['JacDataLoader']


class _InitFunctionWrapper(object):
    def __init__(self, base_seed, fn_init, args, kwargs):
        self._base_seed = base_seed
        self._fn_init = fn_init
        self._args = args
        self._kwargs = kwargs

    def __call__(self, worker_id):
        seed = (self._base_seed + worker_id) % 42964967296
        reset_global_seed(seed)
        if self._fn_init is not None:
            args = self._args[worker_id]
            kwargs = self._kwargs[worker_id]
            self._fn_init(worker_id, *args, **kwargs)


class JacDataLoader(DataLoader):
    """
    A customized dataloader class. It supports an customized initialization function on each worker, as well as
    the initialization of random seed on different workers. It will invoke `jacinle.random.reset_global_seed` to reset
    the random seed upon the initialization of each worker.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, base_seed=None, worker_init_fn=None, worker_init_args=None, worker_init_kwargs=None,
                 **kwargs):

        worker_init_args = worker_init_args if worker_init_args is not None else [tuple() for _ in range(num_workers)]
        worker_init_kwargs = worker_init_kwargs if worker_init_kwargs is not None else [{} for _ in range(num_workers)]

        base_seed = base_seed if base_seed is not None else gen_seed()
        worker_init_fn = _InitFunctionWrapper(base_seed, worker_init_fn, worker_init_args, worker_init_kwargs)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                         num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
                         timeout=timeout, worker_init_fn=worker_init_fn, **kwargs)


if torch.__version__ < '0.3.1':
    from .dataloader_v030 import JacDataLoader

