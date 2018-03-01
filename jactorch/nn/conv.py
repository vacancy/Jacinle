# -*- coding: utf-8 -*-
# File   : conv.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/02/2018
#
# This file is part of Jacinle.

import collections
from itertools import repeat

import torch.nn as nn
import torch.nn.functional as F

from jacinle.utils.enum import JacEnum

__all__ = [
    'ConvPaddingMode', 'ConvBorderMode',
    'Conv1d', 'Conv2d', 'Conv3d',
    'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'ResizeConv1d', 'ResizeConv2d', 'ResizeConv3d'
]


class ConvPaddingMode(JacEnum):
    DEFAULT = 'default'
    VALID = 'valid'
    SAME = 'same'


class ConvBorderMode(JacEnum):
    ZERO = 'zero'
    REFLECT = 'reflect'
    REPLICATE = 'replicate'


class ConvNDBase(nn.Module):
    __nr_dims__ = 1
    __transposed__ = False

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding_mode='default', padding=0, border_mode='zero',
                 output_padding=0, output_border_mode='zero',
                 dilation=1, groups=1, bias=True):

        super().__init__()

        nr_dims = type(self).__nr_dims__
        if not type(self).__transposed__:  # convolution forward
            clz_name = 'Conv{}d'.format(nr_dims)
            self.conv = getattr(nn, clz_name)(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias
            )
        else:
            clz_name = 'ConvTranspose{}d'.format(nr_dims)
            self.conv = getattr(nn, clz_name)(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias,
                output_padding=output_padding
            )

        self.padding_mode = ConvPaddingMode.from_string(padding_mode)
        self.padding = padding
        self.border_mode = ConvBorderMode.from_string(border_mode)
        self.output_padding = output_padding
        self.output_border_mode = ConvBorderMode.from_string(output_border_mode)

        if type(self).__transposed__:
            assert self.border_mode is ConvBorderMode.ZERO, 'Only zero input padding is supported,'
            assert self.output_border_mode is ConvBorderMode.ZERO, 'Only zero output padding is supported.'

    def forward(self, input):
        return self._forward_conv(*self._forward_padding(input))

    def _forward_padding(self, input):
        padding = self._compute_padding(input_size=input.size()[2:])
        if self.border_mode is ConvBorderMode.ZERO:
            return input, padding

        if input.dim() == 4:
            padded = F.pad(input, (padding[1], padding[1], padding[0], padding[0]), mode=self.border_mode.value)
        elif input.dim() == 5:
            padded = F.pad(input, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]),
                           mode=self.border_mode.value)
        else:
            raise ValueError('Only 4D or 5D inputs are supported.')
        conv_padding = self._format_tuple(0)
        return padded, conv_padding

    def _forward_conv(self, padded, extra_padding, **kwargs):
        self.conv.padding = extra_padding
        return self.conv(padded, **kwargs)

    def _compute_padding(self, input_size):
        mode = self.padding_mode
        if mode is ConvPaddingMode.DEFAULT:
            return self._format_tuple(self.padding)
        elif mode is ConvPaddingMode.VALID:
            return self._format_tuple(0)
        elif mode is ConvPaddingMode.SAME:
            kernel_size = self.conv.kernel_size
            assert all(map(lambda x: x % 2 == 1, kernel_size))
            return tuple([k // 2 for k in kernel_size])
        elif mode == ConvPaddingMode.TENSORFLOW:
            raise NotImplementedError()

    def _format_tuple(self, val):
        if isinstance(val, collections.Iterable):
            return tuple(val)
        return tuple(repeat(val, type(self).__nr_dims__))


class Conv1d(ConvNDBase):
    __nr_dims__ = 1


class Conv2d(ConvNDBase):
    __nr_dims__ = 2


class Conv3d(ConvNDBase):
    __nr_dims__ = 3


class ConvTransposeNDBase(ConvNDBase):
    __transposed__ = True

    def forward(self, input, output_size=None, scale_factor=None):
        if output_size is None:
            if scale_factor is not None:
                if isinstance(scale_factor, collections.Sequence):
                    output_size = input.size()[:2] + tuple([s * f for s, f in zip(input.size()[2:], scale_factor)])
                else:
                    output_size = input.size()[:2] + tuple([s * scale_factor for s in input.size()[2:]])
        return self._forward_conv(*self._forward_padding(input), output_size=output_size)


class ConvTranspose1d(ConvTransposeNDBase):
    __nr_dims__ = 1


class ConvTranspose2d(ConvTransposeNDBase):
    __nr_dims__ = 2


class ConvTranspose3d(ConvTransposeNDBase):
    __nr_dims__ = 3


class ResizeConvBase(ConvNDBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding_mode='same', padding=0, border_mode='replicate',
                 dilation=1, groups=1, bias=True,
                 output_size=None, scale_factor=None, resize_mode='nearest'):

        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding_mode=padding_mode, padding=padding, border_mode=border_mode,
                         dilation=dilation, groups=groups, bias=bias)
        self.upsample = nn.Upsample(size=output_size, scale_factor=scale_factor, mode=resize_mode)

    def forward(self, input):
        return super().forward(self.upsample(input))


class ResizeConv1d(ResizeConvBase):
    __nr_dims__ = 1


class ResizeConv2d(ResizeConvBase):
    __nr_dims__ = 2


class ResizeConv3d(ResizeConvBase):
    __nr_dims__ = 3
