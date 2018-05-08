#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : batch.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from copy import deepcopy
from threading import Thread, Event
import traceback

from jacinle.concurrency.event import MTBooleanEvent
from jacinle.logging import get_logger
from jacinle.utils.meta import gofor

from .dataflow import SimpleDataFlowBase

logger = get_logger(__file__)

__all__ = ['BatchDataFlow', 'EpochDataFlow']


def batch_default_filler(buffer, idx, val):
    for k, v in gofor(val):
        if k in buffer:
            buffer[k][idx] = v


class BatchDataFlow(SimpleDataFlowBase):
    _buffer = None
    _cond = None

    _filler_thread = None
    _stop_event = None

    def __init__(self, source, batch_size, sample_dict, filler=batch_default_filler):
        super().__init__()
        self._source = source
        self._batch_size = batch_size
        self._sample_dict = sample_dict
        self._filler = filler

    def _initialize(self):
        self._initialize_buffer()
        self._initialize_filler()

    def _initialize_buffer(self):
        self._buffer = [deepcopy(self._sample_dict) for _ in range(2)]

    def _initialize_filler(self):
        self._cond = [MTBooleanEvent() for _ in range(2)]
        self._stop_event = Event()
        self._filler_thread = Thread(target=self._filler_mainloop, name=str(self) + ':filler', daemon=True)
        self._filler_thread.start()

    def _filler_mainloop(self):
        current = 0
        it = iter(self._source)
        try:
            while True:
                self._cond[current].wait_false()
                for i in range(self._batch_size):
                    self._filler(self._buffer[current], i, next(it))
                self._cond[current].set_true()
                current = 1 - current
        except Exception as e:
            logger.warn('{} got exception {} in filler thread: {}.'.format(type(self), type(e), e))
            traceback.print_exc()
            self._cond[current].set_true()
            self._stop_event.set()

    def _gen(self):
        current = 0
        while True:
            self._cond[current].wait_true()
            if self._stop_event.is_set():
                return
            yield self._buffer[current]
            self._cond[current].set_false()
            current = 1 - current

    def _len(self):
        length = len(self._source)
        return None if length is None else length // self._batch_size


class EpochDataFlow(SimpleDataFlowBase):
    def __init__(self, source, epoch_size):
        self._source = source
        self._source_iter = None
        self._epoch_size = epoch_size

    def _initialize(self):
        self._source_iter = iter(self._source)

    def _gen(self):
        for i in range(self._epoch_size):
            try:
                yield next(self._source_iter)
            except StopIteration:
                return

    def _len(self):
        return self._epoch_size
