# -*- coding: utf8 -*-

import torch.cuda as cuda
import torch.nn as nn

from jactorch.cuda.copy import async_copy_to
from .gather import dict_gather

__all__ = ['UserScatteredDataParallel', 'user_scattered_collate']


class DictGatherDataParallel(nn.DataParallel):
    def gather(self, outputs, output_device):
        return dict_gather(outputs, output_device, dim=self.dim)


class UserScatteredDataParallel(DictGatherDataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        assert len(inputs) == 1
        inputs = inputs[0]
        inputs = _async_copy_stream(inputs, device_ids)
        inputs = [[i] for i in inputs]
        assert len(kwargs) == 0
        kwargs = [{} for _ in range(len(inputs))]

        return inputs, kwargs


def user_scattered_collate(batch):
    return batch


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
