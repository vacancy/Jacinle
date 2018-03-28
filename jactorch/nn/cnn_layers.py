# -*- coding: utf-8 -*-
# File   : cnn_layers.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/02/2018
#
# This file is part of Jacinle.

import torch.nn as nn

from . import conv as conv
from jacinle.utils.enum import JacEnum
from jactorch.nn.quickaccess import get_batcnnorm, get_dropout, get_activation

__all__ = [
    'LinearLayer',
    'Conv1dLayer', 'Conv2dLayer', 'Conv3dLayer',
    'DeconvAlgo', 'Deconv1dLayer', 'Deconv2dLayer', 'Deconv3dLayer'
]


class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, batch_norm=None, dropout=None, bias=None, activation=None):
        if bias is None:
            bias = (batch_norm is None)

        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if batch_norm is not None and batch_norm is not False:
            modules.append(get_batcnnorm(batch_norm, out_features, 1))
        if dropout is not None and dropout is not False:
            modules.append(get_dropout(dropout, 1))
        if activation is not None and activation is not False:
            modules.append(get_activation(activation))
        super().__init__(*modules)


class ConvNDLayerBase(nn.Sequential):
    __nr_dims__ = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding_mode='default', padding=0, border_mode='zero',
                 dilation=1, groups=1,
                 batch_norm=None, dropout=None, bias=None, activation=None):

        if bias is None:
            bias = (batch_norm is None)

        nr_dims = type(self).__nr_dims__
        clz_name = 'Conv{}d'.format(nr_dims)
        modules = [getattr(conv, clz_name)(
            in_channels, out_channels, kernel_size,
            stride=stride, padding_mode=padding_mode, padding=padding, border_mode=border_mode,
            dilation=dilation, groups=groups, bias=bias
        )]

        if batch_norm is not None and batch_norm is not False:
            modules.append(get_batcnnorm(batch_norm, out_channels, nr_dims))
        if dropout is not None and dropout is not False:
            modules.append(get_dropout(dropout, nr_dims))
        if activation is not None and activation is not False:
            modules.append(get_activation(activation))

        super().__init__(*modules)


class Conv1dLayer(ConvNDLayerBase):
    __nr_dims__ = 1


class Conv2dLayer(ConvNDLayerBase):
    __nr_dims__ = 2


class Conv3dLayer(ConvNDLayerBase):
    __nr_dims__ = 3


class DeconvAlgo(JacEnum):
    RESIZECONV = 'resizeconv'
    CONVTRANSPOSE = 'convtranspose'


class _DeconvLayerBase(nn.Module):
    __nr_dims__ = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=None,
                 padding_mode='same', padding=0, border_mode=None,
                 dilation=1, groups=1,
                 output_size=None, scale_factor=None, resize_mode='nearest',
                 batch_norm=None, dropout=None, bias=None, activation=None,
                 algo='resizeconv'):

        super().__init__()

        if bias is None:
            bias = (batch_norm is None)
        self.algo = algo = DeconvAlgo.from_string(algo)

        nr_dims = type(self).__nr_dims__
        if algo is DeconvAlgo.CONVTRANSPOSE:
            clz_name = 'ConvTranspose{}d'.format(nr_dims)
            assert scale_factor is not None
            assert stride is None, 'Can not set strides for Conv-Transpose based Deconv.'
            assert border_mode is None, 'Can not set strides for Conv-Transpose based Deconv.'
            self.deconv = getattr(conv, clz_name)(
                in_channels, out_channels, kernel_size, stride=scale_factor,
                padding_mode=padding_mode, padding=padding, border_mode='zero',
                dilation=dilation, groups=groups, bias=bias
            )
            self.output_size = output_size
            self.scale_factor = scale_factor
        elif algo is DeconvAlgo.RESIZECONV:
            clz_name = 'ResizeConv{}d'.format(nr_dims)
            stride = stride or 1
            border_mode = border_mode or 'replicate'
            self.deconv = getattr(conv, clz_name)(
                in_channels, out_channels, kernel_size, stride=stride,
                padding_mode=padding_mode, padding=padding, border_mode=border_mode,
                dilation=dilation, groups=groups, bias=bias,
                output_size=output_size, scale_factor=scale_factor, resize_mode=resize_mode
            )

        post_modules = []
        if batch_norm is not None and batch_norm is not False:
            post_modules.append(get_batcnnorm(batch_norm, out_channels, nr_dims))
        if dropout is not None and dropout is not False:
            post_modules.append(get_dropout(dropout, nr_dims))
        if activation is not None and activation is not False:
            post_modules.append(get_activation(activation))
        self.post_process = nn.Sequential(*post_modules)

    def forward(self, input, output_size=None):
        if self.algo is DeconvAlgo.CONVTRANSPOSE:
            output_size = output_size if output_size is not None else self.output_size
            return self.post_process(self.deconv(input, output_size, scale_factor=self.scale_factor))
        elif self.algo is DeconvAlgo.RESIZECONV:
            assert output_size is None
            return self.post_process(self.deconv(input))


class Deconv1dLayer(_DeconvLayerBase):
    __nr_dims__ = 1


class Deconv2dLayer(_DeconvLayerBase):
    __nr_dims__ = 2


class Deconv3dLayer(_DeconvLayerBase):
    __nr_dims__ = 3
