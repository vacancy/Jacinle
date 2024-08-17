#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : unsafe_queue.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/17/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import collections
import heapq
import itertools


class Queue(object):
    """A simple queue implementation based on collections.deque."""

    def __init__(self):
        self._queue = collections.deque()

    @property
    def size(self):
        return len(self._queue)

    @property
    def queue(self) -> collections.deque:
        return self._queue

    def put(self, item):
        self._queue.append(item)

    def get(self):
        return self._queue.popleft()

    def empty(self):
        return len(self._queue) == 0


PRIORITY_QUEUE_ITEM_COUNTER = itertools.count()


class PriorityQueue(object):
    """A simple priority queue implementation based on heapq."""

    def __init__(self):
        self._queue = []

    def put(self, item, priority):
        heapq.heappush(self._queue, (priority, next(PRIORITY_QUEUE_ITEM_COUNTER), item))

    def get(self):
        return heapq.heappop(self._queue)[2]

    def empty(self):
        return len(self._queue) == 0
