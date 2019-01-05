#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : counter.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import itertools
import threading
import queue
import collections
import time

__all__ = ['Counter', 'CounterBasedEvent', 'CounterBasedMonitor']


class Counter(object):
    def __init__(self):
        self._cnt = itertools.count()
        self._ref = itertools.count()
        self._iter_cnt = iter(self._cnt)
        self._iter_ref = iter(self._ref)

    def tick(self):
        next(self._iter_cnt)

    def get(self):
        ref = next(self._iter_ref)
        cnt = next(self._iter_cnt)
        return cnt - ref


class CounterBasedEvent(object):
    """Thread-safe counter-based callback invoker. When the counter is incremented, the system will check whether
    the counter has reached a target value. If so, the event will be set."""
    def __init__(self, target, tqdm=None):
        self._cnt = itertools.count()
        self._iter_cnt = iter(self._cnt)

        self._target = target
        self._event = threading.Event()

        self._tick_mutex = threading.Lock()
        self._tqdm = tqdm

    def tick(self):
        with self._tick_mutex:
            return self.__tick()

    def __tick(self):
        value = next(self._iter_cnt)
        if self._tqdm is not None:
            self._tqdm.update(1)
        if value >= self._target:
            self._event.set()
            if self._tqdm is not None:
                self._tqdm.close()
        return value

    def is_set(self):
        return self._event.is_set()

    def clear(self):
        self._event.clear()

    def wait(self, timeout=None):
        return self._event.wait(timeout=timeout)


class CounterBasedMonitor(object):
    _displayer = None

    def __init__(self, counters=None, display_names=None, interval=1, printf=None):
        if counters is None:
            counters = ['DEFAULT']

        self._display_names = display_names
        self._counters = collections.OrderedDict([(n, Counter()) for n in counters])
        self._interval = interval
        self._printf = printf

        if self._printf is None:
            from jacinle.logging import get_logger
            logger = get_logger(__file__)
            self._printf = logger.info

    @property
    def _counter_names(self):
        return list(self._counters.keys())

    def tick(self, name=None):
        if len(self._counter_names) == 1:
            self._counters[self._counter_names[0]].tick()
        else:
            assert name is None, 'Must provide name if there are multiple counters.'
            self._counters[name].tick()

    def start(self):
        self._displayer = threading.Thread(target=self._display_thread, daemon=True)
        self._displayer.start()
        return self

    def _display(self, deltas, interval):
        names = self._display_names or self._counter_names
        if len(names) == 1:
            self._printf('Counter monitor {}: {} ticks/s.'.format(names[0], deltas[0]/interval))
        else:
            log_strs = ['Counter monitor:']
            for n, v in zip(names, deltas):
                log_strs.append('\t{}: {} ticks/s'.format(n, v/interval))
            self._printf('\n'.join(log_strs))

    def _display_thread(self):
        prev = [c.get() for _, c in self._counters.items()]
        while True:
            time.sleep(self._interval)
            curr = [c.get() for _, c in self._counters.items()]
            deltas = [c - p for p, c in zip(prev, curr)]
            prev = curr
            self._display(deltas, self._interval)
