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
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count


class GroupMeters(object):
    def __init__(self):
        self._meters = collections.defaultdict(AverageMeter)

    def reset(self):
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
        return self._meters[name]

    def items(self):
        return self._meters.items()

    @property
    def sum(self):
        return {k: m.sum for k, m in self._meters.items() if m.count > 0}

    @property
    def avg(self):
        return {k: m.avg for k, m in self._meters.items() if m.count > 0}

    @property
    def val(self):
        return {k: m.val for k, m in self._meters.items() if m.count > 0}

    def format(self, caption, values, kv_format, glue):
        meters_kv = self._canonize_values(values)
        log_str = [caption]
        log_str.extend(itertools.starmap(kv_format.format, sorted(meters_kv.items())))
        return glue.join(log_str)

    def format_simple(self, caption, values='avg', compressed=True):
        if compressed:
            return self.format(caption, values, '{}={:4f}', ' ')
        else:
            return self.format(caption, values, '\t{} = {:4f}', '\n')

    def dump(self, filename, values='avg'):
        meters_kv = self._canonize_values(values)
        with open(filename, 'a') as f:
            f.write(io.dumps_json(meters_kv, compressed=False))
            f.write('\n')

    def _canonize_values(self, values):
        if isinstance(values, six.string_types):
            assert values in ('avg', 'val', 'sum')
            meters_kv = getattr(self, values)
        else:
            meters_kv = values
        return meters_kv
