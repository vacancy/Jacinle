#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/05/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections
from itertools import repeat

import torch.nn.functional as F

from jacinle.utils.enum import JacEnum

__all__ = ['ConvPaddingMode', 'ConvBorderMode', 'compute_padding_shape', 'padding_nd']


class ConvPaddingMode(JacEnum):
    DEFAULT = 'default'
    VALID = 'valid'
    SAME = 'same'
    TENSORFLOW = 'tensorflow'


class ConvBorderMode(JacEnum):
    ZERO = 'zero'
    REFLECT = 'reflect'
    REPLICATE = 'replicate'


def _format_tuple(val, arity):
    if isinstance(val, collections.Iterable):
        return tuple(val)
    return tuple(repeat(val, arity))


def compute_padding_shape(input_size, kernel_size, padding, mode):
    mode = ConvPaddingMode.from_string(mode)

    if mode is ConvPaddingMode.DEFAULT:
        return _format_tuple(padding, len(input_size))
    elif mode is ConvPaddingMode.VALID:
        return _format_tuple(0, len(input_size))
    elif mode is ConvPaddingMode.SAME:
        kernel_size = _format_tuple(kernel_size, len(input_size))
        assert all(map(lambda x: x % 2 == 1, kernel_size))
        return tuple([k // 2 for k in kernel_size])
    elif mode == ConvPaddingMode.TENSORFLOW:
        raise NotImplementedError()


def padding_nd(input, kernel_size, padding, padding_mode, border_mode, use_pytorch_padding_mode=False):
    padding_mode = ConvPaddingMode.from_string(padding_mode)
    border_mode = ConvBorderMode.from_string(border_mode)

    padding = compute_padding_shape(input.size()[2:], kernel_size, padding, padding_mode)

    if use_pytorch_padding_mode:
        return input, padding, border_mode.value

    if border_mode is ConvBorderMode.ZERO:
        return input, padding

    if input.dim() == 3:
        padded = F.pad(input, (padding[0], padding[0]), mode=border_mode.value)
    elif input.dim() == 4:
        padded = F.pad(input, (padding[1], padding[1], padding[0], padding[0]), mode=border_mode.value)
    elif input.dim() == 5:
        padded = F.pad(input, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]),
                       mode=border_mode.value)
    else:
        raise ValueError('Only 4D or 5D inputs are supported.')

    conv_padding = _format_tuple(0, input.dim() - 2)
    return padded, conv_padding

