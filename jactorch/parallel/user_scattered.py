#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : user_scattered.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.cuda as cuda
from torch.nn.parallel.data_parallel import DataParallel

from jactorch.cuda.copy import async_copy_to

__all__ = ['UserScatteredDataParallel', 'use_user_scattered']


class UserScatteredDataParallel(DataParallel):
    use_copy_stream = True

    def scatter(self, inputs, kwargs, device_ids):
        return use_user_scattered(inputs, kwargs, device_ids, use_stream=self.use_copy_stream)


def use_user_scattered(inputs, kwargs, device_ids, use_stream=True):
    assert len(inputs) == 1
    inputs = inputs[0]
    if use_stream:
        inputs = _async_copy_stream(inputs, device_ids)
    else:
        inputs = _async_copy(inputs, device_ids)

    inputs = [[i] for i in inputs]
    assert len(kwargs) == 0
    kwargs = [{} for _ in range(len(inputs))]

    return inputs, kwargs


def _async_copy(inputs, device_ids):
    nr_devs = len(device_ids)
    assert type(inputs) in (tuple, list)
    assert len(inputs) == nr_devs

    outputs = []
    for i, dev in zip(inputs, device_ids):
        with cuda.device(dev):
            outputs.append(async_copy_to(i, dev))

    return tuple(outputs)


def _async_copy_stream(inputs, device_ids):
    nr_devs = len(device_ids)
    assert type(inputs) in (tuple, list)
    assert len(inputs) == nr_devs

    outputs = []
    streams = [_get_stream(d) for d in device_ids]
    for i, dev, stream in zip(inputs, device_ids, streams):
        with cuda.device(dev):
            main_stream = cuda.current_stream()
            with cuda.stream(stream):
                outputs.append(async_copy_to(i, dev, main_stream=main_stream))
            main_stream.wait_stream(stream)

    return outputs


"""Adapted from: torch/nn/parallel/_functions.py"""
# background streams used for copying
_streams = None


def _get_stream(device):
    """Gets a background stream for copying between CPU and GPU"""
    global _streams
    if device == -1:
        return None
    if _streams is None:
        _streams = [None] * cuda.device_count()
    if _streams[device] is None: _streams[device] = cuda.Stream(device)
    return _streams[device]
