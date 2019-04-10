#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : device.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os
import collections

from jacinle.utils.enum import JacEnum

__all__ = [
    'DeviceNameFormat', 'canonlize_device_name',
    'parse_devices', 'set_cuda_visible_devices', 'parse_and_set_devices'
]


class DeviceNameFormat(JacEnum):
    TENSORFLOW = 'tensorflow'
    INT = 'int'


def canonlize_device_name(d, format=DeviceNameFormat.INT):
    format = DeviceNameFormat.from_string(format)

    d = d.strip().lower()
    if d == 'cpu':
        if format is DeviceNameFormat.TENSORFLOW:
            return '/cpu:0'
        elif format is DeviceNameFormat.INT:
            return -1

    if d.startswith('gpu'):
        d = d[3:]
    d = int(d)

    if format is DeviceNameFormat.TENSORFLOW:
        return '/gpu:' + str(d)
    elif format is DeviceNameFormat.INT:
        return d


def parse_devices(devs, format=DeviceNameFormat.INT):
    if type(devs) is str:
        devs = devs.split(',')
    else:
        assert isinstance(devs, collections.Sequence)
        if len(devs) == 0:
            devs = devs[0].split(',')

    devs = [canonlize_device_name(d, format) for d in devs]
    return devs


def set_cuda_visible_devices(devs):
    devs = parse_devices(devs, DeviceNameFormat.INT)
    all_gpus = [str(d) for d in devs if d > -1]  # select only GPUs.
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(all_gpus)


def parse_and_set_devices(devs, format=DeviceNameFormat.INT, set_device=True):
    if set_device:
        set_cuda_visible_devices(devs)
    return parse_devices(devs, format)

