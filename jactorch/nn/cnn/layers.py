#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : layers.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.nn as nn

from . import conv as conv
from jacinle.utils.enum import JacEnum
from jactorch.nn.quickaccess import get_batcnnorm, get_dropout, get_activation

__all__ = [
    'LinearLayer', 'MLPLayer',
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

        self.in_features = in_features
        self.out_features = out_features

    @property
    def input_dim(self):
        return self.in_features

    @property
    def output_dim(self):
        return self.out_features

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu', flatten=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []

        nr_hiddens = len(hidden_dims)
        for i in range(nr_hiddens):
            layer = LinearLayer(dims[i], dims[i+1], batch_norm=batch_norm, dropout=dropout, activation=activation)
            modules.append(layer)
        layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        self.mlp = nn.Sequential(*modules)
        self.flatten = flatten

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.mlp(input)


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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.padding = padding
        self.border_mode = border_mode
        self.dilation = dilation
        self.groups = groups

    def reset_parameters(self):
        for module in self.modules():
            if 'Conv' in module.__class__.__name__:
                module.reset_parameters()

    @property
    def input_dim(self):
        return self.in_channels

    @property
    def output_dim(self):
        return self.out_channels


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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.padding = padding
        self.border_mode = border_mode
        self.dilation = dilation
        self.groups = groups

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

        self.output_size = output_size
        self.scale_factor = scale_factor

        post_modules = []
        if batch_norm is not None and batch_norm is not False:
            post_modules.append(get_batcnnorm(batch_norm, out_channels, nr_dims))
        if dropout is not None and dropout is not False:
            post_modules.append(get_dropout(dropout, nr_dims))
        if activation is not None and activation is not False:
            post_modules.append(get_activation(activation))
        self.post_process = nn.Sequential(*post_modules)

    @property
    def input_dim(self):
        return self.in_channels

    @property
    def output_dim(self):
        return self.out_channels

    def reset_parameters(self):
        for module in self.modules():
            if 'Conv' in module.__class__.__name__:
                module.reset_parameters()

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
