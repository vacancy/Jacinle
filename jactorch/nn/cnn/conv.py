#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : conv.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections
import torch.nn as nn
from .functional import ConvPaddingMode, ConvBorderMode, padding_nd

__all__ = [
    'Conv1d', 'Conv2d', 'Conv3d',
    'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'ResizeConv1d', 'ResizeConv2d', 'ResizeConv3d',
    'SequenceConvWrapper'
]


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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.padding_mode = ConvPaddingMode.from_string(padding_mode)
        self.padding = padding
        self.border_mode = ConvBorderMode.from_string(border_mode)
        self.output_padding = output_padding
        self.output_border_mode = ConvBorderMode.from_string(output_border_mode)

        if type(self).__transposed__:
            assert self.border_mode is ConvBorderMode.ZERO, 'Only zero input padding is supported.'
            assert self.output_border_mode is ConvBorderMode.ZERO, 'Only zero output padding is supported.'
        else:
            assert self.output_padding == 0, 'Output padding is only available for transposed convolution.'

    @property
    def input_dim(self):
        return self.in_channels

    @property
    def output_dim(self):
        return self.out_channels

    def forward(self, input):
        # TODO(Jiayuan Mao @ 04/05): evaluate this.
        return self._forward_conv(*self._forward_padding(input))

    def _forward_conv(self, padded, extra_padding, extra_padding_mode=None, **kwargs):
        self.conv.padding = extra_padding
        if extra_padding_mode is not None:
            self.conv.padding_mode = extra_padding_mode
        return self.conv(padded, **kwargs)

    def _forward_padding(self, input):
        use_pytorch_padding_mode = hasattr(self.conv, 'padding_mode')
        return padding_nd(
            input, self.conv.kernel_size, self.padding, self.padding_mode, self.border_mode,
            use_pytorch_padding_mode=use_pytorch_padding_mode
        )


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


class SequenceConvWrapper(nn.Module):
    """
    Wrapper for a sequence of Conv1D layers, support automatic dimension permutation to fit the requirement of
    Conv1D.
    """
    def __init__(self, *modules, batch_first=True):
        super().__init__()
        self.sequential = nn.Sequential(*modules)
        self.batch_first = batch_first

    def forward(self, input):
        assert input.dim() == 3, 'Expect 3-dim input, but got: {}.'.format(input.size())
        if self.batch_first:
            input = input.permute(0, 2, 1)
        else:
            input = input.permute(1, 2, 0)

        input = self.sequential(input)

        if self.batch_first:
            input = input.permute(0, 2, 1)
        else:
            input = input.permute(2, 0, 1)
        return input

