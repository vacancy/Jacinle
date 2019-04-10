#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gradient.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/04/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from .conv import custom_kernel, CustomKernel

__all__ = [
    'ImageGradient', 'image_gradient',
    'Sobel', 'Scharr', 'sobel', 'scharr',
    'Laplacian', 'laplacian'
]


class ImageGradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('kernel_x', torch.tensor([[0.5, 0, -0.5]], dtype=torch.float32))
        self.register_buffer('kernel_y', torch.tensor([[0.5], [0], [-0.5]], dtype=torch.float32))

    def forward(self, image, return_angle=False):
        dx = custom_kernel(image, self.kernel_x)
        dy = custom_kernel(image, self.kernel_y)

        if return_angle:
            return (dx ** 2 + dy ** 2).sqrt(), torch.atan2(dy, dx)
        return (dx ** 2 + dy ** 2).sqrt()


def image_gradient(image, return_angle=False):
    return ImageGradient().to(image.device)(image, return_angle=return_angle)


class SobelBase(nn.Module):
    def __init__(self, kernel_x, kernel_y, norm=1):
        super().__init__()
        assert norm in (1, 2)
        self.norm = norm
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, image, return_angle=False):
        dx = custom_kernel(image, self.kernel_x)
        dy = custom_kernel(image, self.kernel_y)

        if self.norm == 1:
            return 0.5 * (dx.abs() + dy.abs())
        elif self.norm == 2:
            return (dx ** 2 + dy ** 2).sqrt()
        else:
            raise ValueError('Unsupported norm: {}.'.format(self.norm))


class Sobel(SobelBase):
    def __init__(self, kernel_size=3, norm=1):
        assert kernel_size == 3
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        super().__init__(kernel_x, kernel_y, norm)


class Scharr(SobelBase):
    def __init__(self, norm=1):
        kernel_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32)
        kernel_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)
        super().__init__(kernel_x, kernel_y, norm)


def sobel(image, kernel_size=3, norm=1):
    return Sobel(kernel_size, norm).to(image.device)(image)


def scharr(image, norm=1):
    return Scharr(norm).to(image.device)(image)


class Laplacian(CustomKernel):
    def __init__(self):
        super().__init__([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


def laplacian(image):
    return Laplacian().to(image.device)(image)

