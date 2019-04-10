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

from .conv import CustomKernel, MaxPoolingKernelDef

__all__ = ['NormalizedBoxSmooth', 'normalized_box_smooth', 'GaussianSmooth', 'GaussianSmoothTruncated', 'gaussian_smooth', 'gaussian_smooth_truncated', 'MaximumSmooth', 'maximum_smooth']


class NormalizedBoxSmooth(CustomKernel):
    def __init__(self, kernel_size, border_mode='reflect'):
        self.kernel_size = get_2dshape(kernel_size)
        super().__init__(self._gen_kernel(), border_mode=border_mode)

    def _gen_kernel(self):
        kernel = torch.ones(self.kernel_size, dtype=torch.float32)
        kernel /= kernel.sum()
        return kernel


def normalized_box_smooth(image, kernel_size, border_mode='reflect'):
    return NormalizedBoxSmooth(kernel_size, border_mode=border_mode).to(image.device)(image)


class GaussianSmooth(CustomKernel):
    def __init__(self, kernel_size, sigma, border_mode='reflect'):
        assert type(kernel_size) is int, 'GaussianSmooth supports only square kernel.'
        self.kernel_size = kernel_size
        self.sigma = float(sigma)

        super().__init__(self._gen_kernel(), border_mode=border_mode)

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


class GaussianSmoothTruncated(GaussianSmooth):
    def __init__(self, sigma, truncate=4, border_mode='reflect'):
        sigma = float(sigma)
        kernel_size = int(sigma * truncate + 0.5)
        kernel_size = 2 * kernel_size + 1
        super().__init__(kernel_size, sigma, border_mode=border_mode)


def gaussian_smooth(image, kernel_size, sigma, border_mode='reflect'):
    return GaussianSmooth(kernel_size, sigma, border_mode=border_mode).to(image.device)(image)


def gaussian_smooth_truncated(image, sigma, truncate=4, border_mode='reflect'):
    return GaussianSmoothTruncated(sigma, truncate=truncate, border_mode=border_mode).to(image.device)(image)


class MaximumSmooth(CustomKernel):
    def __init__(self, kernel_size, border_mode='reflect'):
        self.kernel_size = kernel_size
        super().__init__(MaxPoolingKernelDef(self.kernel_size), border_mode=border_mode)


def maximum_smooth(image, kernel_size, border_mode='reflect'):
    return MaximumSmooth(kernel_size, border_mode=border_mode).to(image.device)(image)

