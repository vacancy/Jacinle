# -*- coding: utf-8 -*-
# File   : cnn.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/02/2018
#
# This file is part of Jacinle.

import torch.nn as nn

from . import sync_batchnorm as sync_bn


def _get_batcnnorm(bn, nr_features=None, nr_dims=1):
    if isinstance(bn, nn.Module):
        return bn

    assert 1 <= nr_dims <= 3

    if bn in (True, 'async'):
        clz_name = 'BatchNorm{}d'.format(nr_dims)
        return getattr(nn, clz_name)(nr_features)
    elif bn == 'sync':
        clz_name = 'SynchronizedBatchNorm{}d'.format(nr_dims)
        return getattr(sync_bn, clz_name)(nr_features)
    else:
        raise ValueError('Unknown type of batch normalization: {}.'.format(bn))


def _get_dropout(dropout, nr_dims=1):
    if isinstance(dropout, nn.Module):
        return dropout
    if nr_dims == 1:
        return nn.Dropout(nr_dims, True)
    else:
        clz_name = 'Dropout{}d'.format(nr_dims)
        return getattr(nn, clz_name)(dropout)


def _get_activation(act):
    if isinstance(act, nn.Module):
        return act

    assert type(act) is str, 'Unknown type of activation: {}.'.format(act)
    act_lower = act.lower()
    if act_lower == 'relu':
        return nn.ReLU(True)
    elif act_lower == 'sigmoid':
        return nn.Sigmoid()
    elif act_lower == 'tanh':
        return nn.Tanh()
    else:
        try:
            return getattr(nn, act)
        except AttributeError:
            raise ValueError('Unknown activation function: {}.'.format(act))


class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, batch_norm=None, dropout=None, bias=None, activation='relu'):
        if bias is None:
            bias = (batch_norm is None)

        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if batch_norm is not None:
            modules.append(_get_batcnnorm(batch_norm, out_features, 1))
        if dropout is not None:
            modules.append(_get_dropout(dropout, 1))
        if activation is not None:
            modules.append(_get_activation(activation))
        super().__init__(*modules)


class _ConvLayerBase(nn.Sequential):
    __nr_dims__ = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 batch_norm=None, dropout=None, bias=None, activation='relu'):

        if bias is None:
            bias = (batch_norm is None)

        nr_dims = type(self).__nr_dims__
        clz_name = 'Conv{}d'.format(nr_dims)
        modules = [getattr(nn, clz_name)(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )]

        if batch_norm is not None:
            modules.append(_get_batcnnorm(batch_norm, out_channels, nr_dims))
        if dropout is not None:
            modules.append(_get_dropout(dropout, nr_dims))
        if activation is not None:
            modules.append(_get_activation(activation))

        super().__init__(*modules)


class Conv1dLayer(_ConvLayerBase):
    __nr_dims__ = 1


class Conv2dLayer(_ConvLayerBase):
    __nr_dims__ = 2


class Conv3dLayer(_ConvLayerBase):
    __nr_dims__ = 3


class _ConvTransposeLayerBase(nn.Sequential):
    __nr_dims__ = 1

    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, output_padding=0, dilation=1, groups=1,
                 batch_norm=None, dropout=None, bias=None, activation='relu'):

        if bias is None:
            bias = (batch_norm is None)

        nr_dims = type(self).__nr_dims__
        clz_name = 'Conv{}d'.format(nr_dims)
        modules = [getattr(nn, clz_name)(
            input_dim, output_dim, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            dilation=dilation, groups=groups, bias=bias
        )]

        if batch_norm is not None:
            modules.append(_get_batcnnorm(batch_norm, output_dim, nr_dims))
        if dropout is not None:
            modules.append(_get_dropout(dropout, nr_dims))
        if activation is not None:
            modules.append(_get_activation(activation))

        super().__init__(*modules)


class ConvTranspose1dLayer(_ConvTransposeLayerBase):
    __nr_dims__ = 1


class ConvTranspose2dLayer(_ConvTransposeLayerBase):
    __nr_dims__ = 2


class ConvTranspose3dLayer(_ConvTransposeLayerBase):
    __nr_dims__ = 3
