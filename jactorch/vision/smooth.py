#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : smooth.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/04/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import math
import torch

from jacinle.utils.argument import get_2dshape
from .conv import CustomKernel

__all__ = ['NormalizedBoxSmooth', 'normalized_box_smooth', 'GaussianSmooth', 'gaussian_smooth']


class NormalizedBoxSmooth(CustomKernel):
    def __init__(self, kernel_size):
        self.kernel_size = get_2dshape(kernel_size)
        super().__init__(self._gen_kernel())

    def _gen_kernel(self):
        kernel = torch.ones(self.kernel_size, dtype=torch.float32)
        kernel /= kernel.sum()
        return kernel


def normalized_box_smooth(image, kernel_size):
    return NormalizedBoxSmooth(kernel_size)(image)


class GaussianSmooth(CustomKernel):
    def __init__(self, kernel_size, sigma):
        assert type(kernel_size) is int, 'GaussianSmooth supports only square kernel.'
        self.kernel_size = kernel_size
        self.sigma = float(sigma)

        super().__init__(self._gen_kernel())

    def _gen_kernel(self):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(self.kernel_size).float()
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (self.kernel_size - 1) / 2.
        variance = self.sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        Z = (1. / (2. * math.pi * variance))
        gaussian_kernel = Z * torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) /
            (2 * variance)
        )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel /= gaussian_kernel.sum()

        return gaussian_kernel


def gaussian_smooth(image, kernel_size, sigma):
    return GaussianSmooth(kernel_size, sigma)(image)

