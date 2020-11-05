#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : queue.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import queue
import heapq

__all__ = ['ListToFill', 'iter_queue', 'sorted_iter']


class ListToFill(list):
    def __init__(self, nr_target):
        """
        Initialize the target

        Args:
            self: (todo): write your description
            nr_target: (todo): write your description
        """
        super().__init__()
        self._nr_target = nr_target

    def append(self, *args, **kwargs):
        """
        Add a new task to the queue.

        Args:
            self: (todo): write your description
        """
        super().append(*args, **kwargs)
        if len(self) >= self._nr_target:
            raise queue.Full()


def iter_queue(q, total=None):
    """
    Iterate over the queue.

    Args:
        q: (dict): write your description
        total: (todo): write your description
    """
    if total is None:
        while True:
            yield q.get()
    else:
        for i in range(total):
            yield q.get()


def sorted_iter(iter, id_func=None):
    """
    Return an iterator over items in iterable.

    Args:
        iter: (int): write your description
        id_func: (todo): write your description
    """
    if id_func is None:
        id_func = lambda x: x[0]

    current = -1
    buffer = []
    for i, v in enumerate(iter):
        if v is None:
            assert len(buffer) == 0, 'Buffer is not empty when receiving stop signal.'
            break

        idv = id_func(v)
        if idv == current + 1:
            yield idv, v
            current += 1
            while len(buffer) and id_func(buffer[0]) == current + 1:
                ele = heapq.heappop(buffer)
                yield ele[0], ele[2]
                current += 1
        else:
            heapq.heappush(buffer, (idv, i, v))
