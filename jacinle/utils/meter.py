# -*- coding: utf-8 -*-
# File   : meter.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 19/01/2018
# 
# This file is part of Jacinle.

import six
import itertools
import collections

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

    def update(self, updates=None, n=1, **kwargs):
        if updates is None:
            updates = {}
        updates.update(kwargs)
        for k, v in updates.items():
            self._meters[k].update(v, n=n)

    @property
    def avg(self):
        return {k: m.avg for k, m in self._meters.items()}

    @property
    def val(self):
        return {k: m.val for k, m in self._meters.items()}

    def format(self, caption, values, kv_format, glue):
        if isinstance(values, six.string_types):
            assert values in ('avg', 'val')
            meters_kv = getattr(self, values)
        else:
            meters_kv = values
        log_str = [caption]
        log_str.extend(itertools.starmap(kv_format.format, sorted(meters_kv.items())))
        return glue.join(log_str)

    def format_simple(self, caption, values='avg', compressed=True):
        if compressed:
            return self.format(caption, values, '{}={:4f}', ' ')
        else:
            return self.format(caption, values, '\t{} = {:4f}', '\n')
