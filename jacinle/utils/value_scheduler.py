#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : value_scheduler.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['ValueScheduler', 'MonotonicSchedulerBase', 'LinearScheduler']


class ValueScheduler(object):
    """Essentially, define y = f(x), where x is a discrete, bounded variable."""
    def get(self, x):
        """
        Returns the value of x.

        Args:
            self: (todo): write your description
            x: (int): write your description
        """
        raise NotImplementedError()


class ConstantScheduler(ValueScheduler):
    def __init__(self, value):
        """
        Initialize the value

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        self.value = value

    def get(self, x):
        """
        Returns the value of x.

        Args:
            self: (todo): write your description
            x: (int): write your description
        """
        return self.value


class MonotonicSchedulerBase(ValueScheduler):
    def __init__(self, begin, begin_value, end, end_value):
        """
        Initialize start_value.

        Args:
            self: (todo): write your description
            begin: (todo): write your description
            begin_value: (float): write your description
            end: (int): write your description
            end_value: (todo): write your description
        """
        super().__init__()
        self.begin = begin
        self.begin_value = begin_value
        self.end = end
        self.end_value = end_value


class LinearScheduler(MonotonicSchedulerBase):
    def get(self, x):
        """
        Return the value at x.

        Args:
            self: (todo): write your description
            x: (int): write your description
        """
        if x < self.begin:
            return self.begin_value
        elif x > self.end:
            return self.end_value
        return self.begin_value + (self.end_value - self.begin_value) / (self.end - self.begin) * (x - self.begin)

