# -*- coding: utf-8 -*-
# File   : dataloader.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/03/2018
#
# This file is part of Jacinle.

import threading
import queue
import multiprocessing

from torch.utils.data.dataloader import _worker_loop, _pin_memory_loop, default_collate, DataLoader, DataLoaderIter

from jacinle.random import reset_global_seed, gen_rng

__all__ = ['JacDataLoader', 'JacDataLoaderIter']


def _worker_loop_seed(worker_id, dataset, index_queue, data_queue, collate_fn, seed, init_func, init_args, init_kwargs):
    reset_global_seed(seed)
    if init_func is not None:
        init_func(worker_id, *init_args, **init_kwargs)
    _worker_loop(dataset, index_queue, data_queue, collate_fn)


class JacDataLoaderIter(DataLoaderIter):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.done_event = threading.Event()

        self.init_func = loader.init_func
        self.init_args = loader.init_args
        self.init_kwargs = loader.init_kwargs

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
                          self.init_func, self.init_args[i], self.init_kwargs[i]))
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
            if self.init_func is not None:
                self.init_func(-1, *self.init_args, **self.init_kwargs)


class JacDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False, seed=None,
                 init_func=None, init_args=None, init_kwargs=None):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                         num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last)
        self.init_func = init_func
        self.init_args = init_args
        self.init_kwargs = init_kwargs

        if num_workers > 0:
            self.seed_generator = gen_rng(seed)
            self.init_args = init_args if init_args is not None else [tuple() for _ in range(num_workers)]
            self.init_kwargs = init_kwargs if init_kwargs is not None else [{} for _ in range(num_workers)]
        else:
            self.init_args = init_args if init_args is not None else tuple()
            self.init_kwargs = init_kwargs if init_kwargs is not None else {}

    def __iter__(self):
        return JacDataLoaderIter(self)

    def gen_seeds(self):
        assert self.num_workers > 0
        return self.seed_generator.randint(4294967296, size=self.num_workers).tolist()
