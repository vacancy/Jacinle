#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dataloader.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import threading
import multiprocessing
import torch
from torch.utils.data.dataloader import DataLoader, default_collate

from jacinle.random import reset_global_seed, gen_seed

__all__ = ['JacDataLoader', 'JacDataLoaderMultiGPUWrapper', 'DataLoaderPipeMaster', 'DataLoaderPipeSlave']


class DataLoaderPipeMaster(object):
    def __init__(self, nr_workers):
        """
        Initialize workers. workers.

        Args:
            self: (todo): write your description
            nr_workers: (todo): write your description
        """
        self.nr_workers = nr_workers
        self.queues = [multiprocessing.Queue() for _ in range(self.nr_workers)]

    def send(self, data):
        """
        Send data to the queue.

        Args:
            self: (todo): write your description
            data: (todo): write your description
        """
        for q in self.queues:
            q.put_nowait(data)


class DataLoaderPipeSlave(object):
    def __init__(self, on_recv_func):
        """
        Initialize the queue.

        Args:
            self: (todo): write your description
            on_recv_func: (todo): write your description
        """
        self.on_recv_func = on_recv_func
        self.queue = None
        self.thread = None

    def worker_init(self, queue):
        """
        Initialize the queue.

        Args:
            self: (todo): write your description
            queue: (todo): write your description
        """
        self.queue = queue
        self.thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.thread.start()

    def recv_loop(self):
        """
        Receive loop.

        Args:
            self: (todo): write your description
        """
        while True:
            data = self.queue.get()
            self.on_recv_func(data)


class _InitFunctionWrapper(object):
    def __init__(self, base_seed, fn_init, args, kwargs, pipe_master, fn_recv):
        """
        Initialize the pipe.

        Args:
            self: (todo): write your description
            base_seed: (str): write your description
            fn_init: (int): write your description
            pipe_master: (todo): write your description
            fn_recv: (int): write your description
        """
        self._base_seed = base_seed
        self._fn_init = fn_init
        self._args = args
        self._kwargs = kwargs
        self._pipe_master = pipe_master
        self._fn_recv = fn_recv

        self._pipe_recv = None
        if self._fn_recv is not None:
            self._pipe_recv = DataLoaderPipeSlave(self._fn_recv)

    def __call__(self, worker_id):
        """
        Initialize the worker.

        Args:
            self: (todo): write your description
            worker_id: (str): write your description
        """
        seed = (self._base_seed + worker_id) % 42964967296
        reset_global_seed(seed)

        if self._fn_init is not None:
            args = self._args[worker_id]
            kwargs = self._kwargs[worker_id]
            self._fn_init(worker_id, *args, **kwargs)

        if self._fn_recv is not None:
            if self._pipe_master is not None and len(self._pipe_master.queues) > 0:
                self._fn_recv.worker_init(self._pipe_master.queues[worker_id])


class JacDataLoader(DataLoader):
    """
    A customized dataloader class. It supports an customized initialization function on each worker, as well as
    the initialization of random seed on different workers. It will invoke `jacinle.random.reset_global_seed` to reset
    the random seed upon the initialization of each worker.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, base_seed=None, worker_init_fn=None, worker_init_args=None, worker_init_kwargs=None,
                 worker_recv_fn=None, **kwargs):
        """
        Initialize the worker.

        Args:
            self: (todo): write your description
            dataset: (todo): write your description
            batch_size: (int): write your description
            shuffle: (bool): write your description
            sampler: (todo): write your description
            batch_sampler: (todo): write your description
            num_workers: (int): write your description
            collate_fn: (todo): write your description
            default_collate: (str): write your description
            pin_memory: (str): write your description
            drop_last: (todo): write your description
            timeout: (int): write your description
            base_seed: (str): write your description
            worker_init_fn: (str): write your description
            worker_init_args: (dict): write your description
            worker_init_kwargs: (dict): write your description
            worker_recv_fn: (todo): write your description
        """

        worker_init_args = worker_init_args if worker_init_args is not None else [tuple() for _ in range(num_workers)]
        worker_init_kwargs = worker_init_kwargs if worker_init_kwargs is not None else [{} for _ in range(num_workers)]

        base_seed = base_seed if base_seed is not None else gen_seed()
        self.worker_recv_fn = worker_recv_fn
        if worker_recv_fn is not None:
            self.pipe_master = DataLoaderPipeMaster(num_workers)
        else:
            self.pipe_master = None

        worker_init_fn = _InitFunctionWrapper(
            base_seed, worker_init_fn, worker_init_args, worker_init_kwargs,
            self.pipe_master, DataLoaderPipeSlave(worker_recv_fn)
        )
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                         num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
                         timeout=timeout, worker_init_fn=worker_init_fn, **kwargs)

    def send_to_worker(self, data):
        """
        Send data to the worker.

        Args:
            self: (todo): write your description
            data: (todo): write your description
        """
        self.worker_recv_fn(data)
        if self.num_workers > 0:
            self.pipe_master.send(data)


class JacDataLoaderMultiGPUWrapper(object):
    def __init__(self, dataloader, gpus):
        """
        Initialize the gpus.

        Args:
            self: (todo): write your description
            dataloader: (todo): write your description
            gpus: (todo): write your description
        """
        self.dataloader = dataloader
        self.gpus = gpus
        self.gpu_parallel = len(gpus) > 1

    @property
    def unwrapped(self):
        """
        Returns the wrapped wrapped datalerer.

        Args:
            self: (todo): write your description
        """
        return self.dataloader

    def __iter__(self):
        """
        Yield an iterator.

        Args:
            self: (todo): write your description
        """
        it = iter(self.dataloader)
        while True:
            gpu_data = list()
            for i in range(len(self.gpus)):
                try:
                    gpu_data.append(next(it))
                except StopIteration:
                    break
            if self.gpu_parallel:
                yield gpu_data
            else:
                yield gpu_data[0]

    def __len__(self):
        """
        The length of the length.

        Args:
            self: (todo): write your description
        """
        return len(self.dataloader)

