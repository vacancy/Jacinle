#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : inference.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time
import queue
import threading
import contextlib

import torch

from jacinle.concurrency.future import FutureResult
from jacinle.utils.meta import map_exec_method
from jactorch.utils.meta import as_numpy, as_tensor
from jacnp.batch import batchify, unbatchify

__all__ = ['ModelInferencer', 'AsyncInferenceTask', 'AsyncModelInferencer', 'BatchedAsyncModelInferencer']


class AsyncInferenceTask(object):
    __slots__ = ('future', 'feed_dict')

    def __init__(self, feed_dict, future=None):
        self.feed_dict = feed_dict
        if future is None:
            future = FutureResult()
        self.future = future

    def get_result(self):
        return self.future.get()

    def put_result(self, result):
        return self.future.put(result)


class ModelInferencer(object):
    def __init__(self, model):
        self._model = model

    @contextlib.contextmanager
    def activate(self):
        self.initialize()
        yield self
        self.finalize()

    def initialize(self):
        pass

    def finalize(self):
        pass

    def inference(self, feed_dict):
        return self._inference_model(feed_dict)

    def _inference_model(self, feed_dict):
        feed_dict = as_tensor(feed_dict)
        with torch.no_grad():
            return as_numpy(self._model(feed_dict))


class AsyncModelInferencer(ModelInferencer):
    def __init__(self, model, nr_workers=1):
        super().__init__(model)
        self._nr_workers = nr_workers
        self._task_queue = None
        self._workers = []

    def initialize(self):
        assert len(self._workers) == 0

        self._task_queue = queue.Queue()
        for rank in range(self._nr_workers):
            th = threading.Thread(target=self._mainloop_worker, args=(rank, ))
            th.start()
            self._workers.append(th)

    def finalize(self):
        if len(self._workers) == 0:
            return

        for rank in range(self._nr_workers):
            self._task_queue.put(None)
        map_exec_method('join', self._workers)

    def _mainloop_worker(self, rank):
        while True:
            task = self._task_queue.get()
            if task is None:
                break
            task.put_result(self._inference_model(task.feed_dict))

    def inference(self, feed_dict, future=None):
        task = AsyncInferenceTask(feed_dict, future=future)
        self._task_queue.put(task)
        return task


class BatchedAsyncModelInferencer(AsyncModelInferencer):
    def __init__(self, model, nr_workers=1, batch_size=8, latency=10):
        super().__init__(model, nr_workers=nr_workers)
        self._batch_size = batch_size
        self._latency = latency / 1000

    def _mainloop_worker(self, rank):
        while True:
            tasks = []
            stop_signal = False

            last_time = time.time() + self._latency
            for i in range(self._batch_size):
                if len(tasks) > 0:
                    task = self._task_queue.get(timeout=last_time - time.time())
                else:
                    task = self._task_queue.get()
                if task is None:
                    stop_signal = True
                    break
                else:
                    tasks.append(task)

            if len(tasks):
                batched_feed = batchify([t.feed_dict for t in tasks])
                outputs = unbatchify(self._inference_model(batched_feed))
                for t, o in zip(tasks, outputs):
                    t.put_result(o)

            if stop_signal:
                break
