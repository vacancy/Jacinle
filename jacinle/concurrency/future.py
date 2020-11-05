#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : future.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import threading

__all__ = ['FutureResult']


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        """
        Initialize the thread.

        Args:
            self: (todo): write your description
        """
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        """
        Put the result to the queue.

        Args:
            self: (todo): write your description
            result: (todo): write your description
        """
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        """
        Get the result of the job.

        Args:
            self: (todo): write your description
        """
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res
