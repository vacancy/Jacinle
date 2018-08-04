#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : coord_conv.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
CoordConv introduced in:
"An intriguing failing of convolutional neural networks and the CoordConv solution".
https://arxiv.org/pdf/1807.03247.pdf

Codes are adapted from https://github.com/mkocabas/CoordConv-pytorch.
"""

import torch
import torch.nn as nn
import jactorch

__all__ = ['CoordConv']


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, *args, use_radius=False, **kwargs):
        super().__init__()

        self.addcoords = _AddCoords(use_radius=use_radius)
        in_channels += self.addcoords.extra_channels
        self.conv = nn.Conv2d(in_channels, out_channels, *args, **kwargs)

    def forward(self, x):
        f = self.addcoords(x)
        f = self.conv(f)
        return f


class _AddCoords(nn.Module):
    def __init__(self, use_radius=False):
        super().__init__()

        self.use_radius = use_radius
        self.extra_channels = 3 if self.use_radius else 2

    def forward(self, input):
        batch_size, _, h, w = input.size()

        def gen(length):
            return -1 + torch.arange(length, dtype=input.dtype, device=input.device) / (length - 1) * 2

        results = [input]
        with torch.no_grad():
            x_coords = gen(w).view(1, 1, 1, w).expand((batch_size, 1, h, w))
            y_coords = gen(h).view(1, 1, h, 1).expand((batch_size, 1, h, w))
            results.extend([x_coords, y_coords])
            if self.use_radius:
                radius = torch.sqrt(torch.pow(x_coords - 0.5, 2) + torch.pow(y_coords - 0.5, 2))
                results.append(radius)

        return torch.cat(results, dim=1)

