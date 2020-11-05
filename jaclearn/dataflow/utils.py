#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.


import queue
import threading

from itertools import repeat, cycle as cached_cycle
from itertools import takewhile, dropwhile, filterfalse
from itertools import chain
from itertools import starmap
from itertools import islice
from itertools import tee

from jacinle.utils.meta import map_exec

from .dataflow import SimpleDataFlowBase, ProxyDataFlowBase


__all__ = [
    'cycle', 'cycle_n', 'cached_cycle', 'repeat', 'repeat_n',
    'chain',
    'takewhile', 'dropwhile',
    'filter', 'filtertrue', 'filterfalse',
    'map', 'starmap', 'ssmap',
    'islice', 'truncate',
    'tee',
    'MapDataFlow', 'DataFlowMixer'
]

map = map
filter = filter
filtertrue = filter
repeat_n = repeat
truncate = islice


# implement cycle self, without any cache
def cycle(iterable, times=None):
    """
    Iterate over a given iterable times.

    Args:
        iterable: (todo): write your description
        times: (list): write your description
    """
    if times is None:
        while True:
            for v in iterable:
                yield v
    else:
        for i in range(times):
            for v in iterable:
                yield v

cycle_n = cycle


def ssmap(function, iterable):
    """
    Yields the given iterable.

    Args:
        function: (todo): write your description
        iterable: (todo): write your description
    """
    for args in iterable:
        yield function(**args)


class MapDataFlow(ProxyDataFlowBase):
    def __init__(self, other, map_func=None):
        """
        Initialize the map.

        Args:
            self: (todo): write your description
            other: (todo): write your description
            map_func: (todo): write your description
        """
        super().__init__(other)
        self.__map_func = map_func

    def _map(self, data):
        """
        Apply a function to the map.

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        return self.__map_func(data)

    def _gen(self):
        """
        Yield a generator that yields the results of this method.

        Args:
            self: (todo): write your description
        """
        for data in self.unwrapped:
            yield self._map(data)


class DataFlowMixer(SimpleDataFlowBase):
    def __init__(self, dataflows, buflen=None):
        """
        Initialize the queue.

        Args:
            self: (todo): write your description
            dataflows: (todo): write your description
            buflen: (int): write your description
        """
        if buflen is None:
            buflen = len(dataflows)
        self._dataflows = dataflows
        self._queue = queue.Queue(maxsize=buflen)
        self._stop_signal = threading.Event()
        self._comsumers = []

    def _initialize(self):
        """
        Initialize the consumer.

        Args:
            self: (todo): write your description
        """
        self._consumers = [
            threading.Thread(target=self._consumer, args=(ind, df), daemon=True)
            for ind, df in enumerate(self._dataflows)]
        map_exec(threading.Thread.start, self._consumers)

    def _finalize(self):
        """
        Finalize the queue.

        Args:
            self: (todo): write your description
        """
        self._stop_signal.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        map_exec(threading.Thread.join, self._consumers)

    def _consumer(self, ind, df):
        """
        Stop consumer.

        Args:
            self: (todo): write your description
            ind: (todo): write your description
            df: (todo): write your description
        """
        for data in df:
            self._queue.put(self._wrapper(data, ind))
            if self._stop_signal.is_set():
                break

    def _gen(self):
        """
        Generator that yields the queue.

        Args:
            self: (todo): write your description
        """
        while True:
            yield self._queue.get()

    def _wrapper(self, data, ind):
        """
        Decorator for function

        Args:
            self: (todo): write your description
            data: (todo): write your description
            ind: (todo): write your description
        """
        return data, ind
