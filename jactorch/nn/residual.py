#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : residual.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/10/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from .container import AutoResetParametersMixin
from .rnn_utils import rnn_with_length
from .normalization import LayerNorm

__all__ = ['ResidualConvBlock', 'ResidualConvBottleneck', 'ResidualLinear', 'ResidualGRU']


def _conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResidualConvBlock(nn.Module, AutoResetParametersMixin):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResidualConvBottleneck(nn.Module, AutoResetParametersMixin):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResidualLinear(nn.Module, AutoResetParametersMixin):
    def __init__(self, hidden_dim, norm1=None, norm2=None):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = norm1
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = norm2
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        f = self.linear1(input)
        if self.norm1 is not None:
            f = self.norm1(f)
        f = self.relu(f)
        f = self.linear2(f)
        if self.norm2 is not None:
            f = self.norm2(f)
        f = f + input
        f = self.relu(f)
        return f


class ResidualGRU(nn.Module, AutoResetParametersMixin):
    def __init__(self, hidden_dim, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False,
                 layer_norm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.real_hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        self.num_layers = num_layers
        self.real_num_layers = num_layers * 2 if bidirectional else num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.rnns = nn.ModuleList([
            nn.GRU(self.hidden_dim, self.real_hidden_dim, 1, bias=bias, batch_first=False, dropout=dropout,
                   bidirectional=bidirectional)
            for _ in range(num_layers)
        ])
        self.layer_norms = None
        if layer_norm:
            self.layer_norms = nn.ModuleList([
                LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ])

    def forward(self, input, input_lengths, initial_states=None):
        if self.batch_first:
            input = input.transpose(0, 1)

        if initial_states is None:
            batch_size = input.size(1)
            state_shape = (self.real_num_layers, batch_size, self.hidden_dim)
            initial_states = torch.zeros(state_shape, device=input.device)

        f = input
        for i in range(self.num_layers):
            f_input = f
            f_state = initial_states[2*i:2*i+2] if self.bidirectional else initial_states[i:i+1]
            # TODO(Jiayuan Mao @ 05/08): accelerate this by pre-sort the sequences.
            f = rnn_with_length(self.rnns[i], f, input_lengths, initial_states=f_state,
                                batch_first=False, sorted=False)
            if self.layer_norms is not None:
                f = self.layer_norms[i](f)
            f = f + f_input

        if self.batch_first:
            f = f.transpose(0, 1)
        return f

