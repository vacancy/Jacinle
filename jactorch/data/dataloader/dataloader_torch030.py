#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dataloader_torch030.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/01/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import threading
import queue
import multiprocessing

try:
    from torch.utils.data.dataloader import _worker_loop, _pin_memory_loop, default_collate, DataLoader, DataLoaderIter
except ImportError:
    DataLoader = object
    DataLoaderIter = object

from jacinle.random import reset_global_seed, gen_rng

__all__ = ['JacDataLoader', 'JacDataLoaderIter']


def _worker_loop_seed(worker_id, dataset, index_queue, data_queue, collate_fn, seed, worker_init_fn, worker_init_args, worker_init_kwargs):
    reset_global_seed(seed)
    if worker_init_fn is not None:
        worker_init_fn(worker_id, *worker_init_args, **worker_init_kwargs)
    _worker_loop(dataset, index_queue, data_queue, collate_fn)


class JacDataLoaderIter(DataLoaderIter):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.done_event = threading.Event()

        self.worker_init_fn = loader.worker_init_fn
        self.worker_init_args = loader.worker_init_args
        self.worker_init_kwargs = loader.worker_init_kwargs

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.index_queue = multiprocessing.SimpleQueue()
            self.data_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.seeds = loader.gen_seeds()
            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop_seed,
                    args=(i, self.dataset, self.index_queue, self.data_queue, self.collate_fn, self.seeds[i],
                          self.worker_init_fn, self.worker_init_args[i], self.worker_init_kwargs[i]))
                for i in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            if self.pin_memory:
                in_data = self.data_queue
                self.data_queue = queue.Queue()
                self.pin_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(in_data, self.data_queue, self.done_event))
                self.pin_thread.daemon = True
                self.pin_thread.start()

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()
        else:
            if self.worker_init_fn is not None:
                self.worker_init_fn(-1, *self.worker_init_args, **self.worker_init_kwargs)


class JacDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False, seed=None,
                 worker_init_fn=None, worker_init_args=None, worker_init_kwargs=None):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                         num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last)
        self.worker_init_fn = worker_init_fn
        self.worker_init_args = worker_init_args
        self.worker_init_kwargs = worker_init_kwargs

        if num_workers > 0:
            self.seed_generator = gen_rng(seed)
            self.worker_init_args = worker_init_args if worker_init_args is not None else [tuple() for _ in range(num_workers)]
            self.worker_init_kwargs = worker_init_kwargs if worker_init_kwargs is not None else [{} for _ in range(num_workers)]
        else:
            self.worker_init_args = worker_init_args if worker_init_args is not None else tuple()
            self.worker_init_kwargs = worker_init_kwargs if worker_init_kwargs is not None else {}

    def __iter__(self):
        return JacDataLoaderIter(self)

    def gen_seeds(self):
        assert self.num_workers > 0
        return self.seed_generator.randint(4294967296, size=self.num_workers).tolist()
