#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : device.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Uiltity functions to parse CPU/CUDA device strings."""

import os
import collections
from typing import Union, Sequence, List

from jacinle.utils.enum import JacEnum

__all__ = [
    'DeviceNameFormat', 'canonlize_device_name',
    'parse_devices', 'set_cuda_visible_devices', 'parse_and_set_devices'
]


class DeviceNameFormat(JacEnum):
    """The target fmt of device names. Supported formats are:

    - ``DeviceNameFormat.INT``: integer, e.g., ``0``, ``1``, ``2``, etc.
    - ``DeviceNameFormat.TENSORFLOW``: TensorFlow-style device name, e.g., ``/cpu:0``, ``/gpu:1``, etc.
    """
    TENSORFLOW = 'tensorflow'
    INT = 'int'


def canonlize_device_name(d: str, fmt: Union[str, DeviceNameFormat] = DeviceNameFormat.INT) -> Union[str, int]:
    """Convert a device name to a canonical format.

    Args:
        d: the device name to be converted. The string can be either: ``cpu``, ``gpu0``, or ``0``.
        fmt: the target format.

    Returns:
        the canonical device name. If the target format is ``DeviceNameFormat.INT``, the return value is an integer.
        When d is ``cpu``, the return value is ``-1``.
        If the target format is ``DeviceNameFormat.TENSORFLOW``, the return value is a string: e.g., ``/cpu:0``, ``/gpu:1``, etc.
    """
    fmt = DeviceNameFormat.from_string(fmt)

    d = d.strip().lower()
    if d == 'cpu':
        if fmt is DeviceNameFormat.TENSORFLOW:
            return '/cpu:0'
        elif fmt is DeviceNameFormat.INT:
            return -1
        else:
            raise ValueError(f'Unknown device name format: {fmt}.')

    if d.startswith('gpu'):
        d = d[3:]
    d = int(d)

    if fmt is DeviceNameFormat.TENSORFLOW:
        return '/gpu:' + str(d)
    elif fmt is DeviceNameFormat.INT:
        return d
    else:
        raise ValueError(f'Unknown device name format: {fmt}.')


def parse_devices(devs: Union[str, Sequence[str]], fmt: Union[str, DeviceNameFormat] = DeviceNameFormat.INT) -> List[Union[str, int]]:
    """Parse a input list of strings or a single comma-separated string into a list of device names.

    Args:
        devs: the input device list.
        fmt: the target format.

    Returns:
        the parsed device list.
    """
    if type(devs) is str:
        devs = devs.split(',')
    else:
        assert isinstance(devs, collections.Sequence)
        if len(devs) == 1:
            devs = devs[0].split(',')

    devs = [canonlize_device_name(d, fmt) for d in devs]
    return devs


def set_cuda_visible_devices(devs: Union[str, Sequence[str]]):
    """Set the CUDA_VISIBLE_DEVICES environment variable with a single comma-separated string or a list of strings.

    Args:
        devs: the input device list. Either a single comma-separated string or a list of strings.
    """
    devs = parse_devices(devs, DeviceNameFormat.INT)
    all_gpus = [str(d) for d in devs if d > -1]  # select only GPUs.
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(all_gpus)


def parse_and_set_devices(devs: Union[str, Sequence[str]], fmt: Union[str, DeviceNameFormat] = DeviceNameFormat.INT, set_device: bool = True):
    """Parse a input list of strings or a single comma-separated string into a list of device names. When ``set_device`` is True,
    the CUDA_VISIBLE_DEVICES environment variable will be set accordingly.

    Args:
        devs: the input device list.
        fmt: the target format.
        set_device: whether to set the CUDA_VISIBLE_DEVICES environment variable.

    Returns:
        the parsed device list.
    """
    if set_device:
        set_cuda_visible_devices(devs)
    return parse_devices(devs, fmt)

