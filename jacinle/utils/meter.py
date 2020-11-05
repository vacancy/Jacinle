#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : meter.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import six
import itertools
import collections

import jacinle.io as io

from .meta import map_exec


class AverageMeter(object):
    """Computes and stores the average and current value"""
    val = 0
    avg = 0
    sum = 0
    count = 0
    tot_count = 0

    def __init__(self):
        """
        Reset the internal state.

        Args:
            self: (todo): write your description
        """
        self.reset()
        self.tot_count = 0

    def reset(self):
        """
        Reset the state.

        Args:
            self: (todo): write your description
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the value.

        Args:
            self: (todo): write your description
            val: (float): write your description
            n: (array): write your description
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count


class GroupMeters(object):
    def __init__(self):
        """
        Initialize meters.

        Args:
            self: (todo): write your description
        """
        self._meters = collections.defaultdict(AverageMeter)

    def reset(self):
        """
        Reset the map.

        Args:
            self: (todo): write your description
        """
        map_exec(AverageMeter.reset, self._meters.values())

    def update(self, updates=None, value=None, n=1, **kwargs):
        """
        Example:
            >>> meters.update(key, value)
            >>> meters.update({key1: value1, key2: value2})
            >>> meters.update(key1=value1, key2=value2)
        """
        if updates is None:
            updates = {}
        if updates is not None and value is not None:
            updates = {updates: value}
        updates.update(kwargs)
        for k, v in updates.items():
            self._meters[k].update(v, n=n)

    def __getitem__(self, name):
        """
        Get a named item by name.

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        return self._meters[name]

    def items(self):
        """
        Return all items.

        Args:
            self: (todo): write your description
        """
        return self._meters.items()

    @property
    def sum(self):
        """
        Return the sum of all metrics.

        Args:
            self: (todo): write your description
        """
        return {k: m.sum for k, m in self._meters.items() if m.count > 0}

    @property
    def avg(self):
        """
        A dictionary of the metric values.

        Args:
            self: (todo): write your description
        """
        return {k: m.avg for k, m in self._meters.items() if m.count > 0}

    @property
    def val(self):
        """
        Return a dict with all the values in this object.

        Args:
            self: (todo): write your description
        """
        return {k: m.val for k, m in self._meters.items() if m.count > 0}

    def format(self, caption, values, kv_format, glue):
        """
        Formats the formatted as a string.

        Args:
            self: (todo): write your description
            caption: (str): write your description
            values: (str): write your description
            kv_format: (str): write your description
            glue: (todo): write your description
        """
        meters_kv = self._canonize_values(values)
        log_str = [caption]
        log_str.extend(itertools.starmap(kv_format.format, sorted(meters_kv.items())))
        return glue.join(log_str)

    def format_simple(self, caption, values='avg', compressed=True):
        """
        Format a formatted representation of a simple representation.

        Args:
            self: (todo): write your description
            caption: (str): write your description
            values: (array): write your description
            compressed: (bool): write your description
        """
        if compressed:
            return self.format(caption, values, '{}={:4f}', ' ')
        else:
            return self.format(caption, values, '\t{} = {:4f}', '\n')

    def dump(self, filename, values='avg'):
        """
        Writes metrics to a file.

        Args:
            self: (todo): write your description
            filename: (str): write your description
            values: (str): write your description
        """
        meters_kv = self._canonize_values(values)
        with open(filename, 'a') as f:
            f.write(io.dumps_json(meters_kv, compressed=False))
            f.write('\n')

    def _canonize_values(self, values):
        """
        Return a canonical version of values.

        Args:
            self: (todo): write your description
            values: (str): write your description
        """
        if isinstance(values, six.string_types):
            assert values in ('avg', 'val', 'sum')
            meters_kv = getattr(self, values)
        else:
            meters_kv = values
        return meters_kv
