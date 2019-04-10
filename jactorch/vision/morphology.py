#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : morphology.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/04/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import jactorch

from jacinle.utils.argument import get_2dshape
from jacinle.utils.enum import JacEnum
from .conv import custom_kernel

__all__ = [
    'MorphologyKernelType', 'get_morphology_kernel', 'MorphologyOp',
    'Erosion', 'Dilation', 'Opening', 'Closing', 'MorphologicalGradient', 'TopHat', 'BlackHat',
    'erode', 'dilate', 'open', 'close', 'morph_grad', 'top_hat', 'black_hat'
]


class MorphologyKernelType(JacEnum):
    RECT = 'rect'
    ELLIPSE = 'ellipse'
    CROSS = 'cross'


def get_morphology_kernel(shape, kernel_size):
    shape = MorphologyKernelType.from_string(shape)
    kernel_size = get_2dshape(kernel_size)

    if shape is MorphologyKernelType.RECT:
        return torch.ones(kernel_size, dtype=torch.float32)
    elif shape is MorphologyKernelType.CROSS:
        kernel = torch.zeros(kernel_size, dtype=torch.float32)
        kernel[kernel_size[0] // 2, :] = 1
        kernel[:, kernel_size[1] // 2] = 1
        return kernel
    elif shape is MorphologyKernelType.ELLIPSE:
        kernel = torch.zeros(kernel_size, dtype=torch.float32)
        r, c = kernel_size[0] // 2, kernel_size[1] // 2
        inv_r2 = 1 / (r * r) if r != 0 else 0
        for i in range(kernel_size[0]):
            j1, j2 = 0, 0
            dy = i - r
            if abs(dy) <= r:
                dx = c * sqrt((r * r - dy * dy) * inv_r2)
                j1 = max(c - dx, 0)
                j2 = min(c + dx + 1, kernel_size[1])
            kernel[i, j1:j2] = 1
        return kernel


def _cvt_morphology_kernel(kernel):
    tot_size = kernel.size(0) * kernel.size(1)
    indices = torch.arange(tot_size)
    flatten_kernel = kernel.view(-1)

    new_kernel = torch.zeros((tot_size, tot_size), dtype=torch.float32)
    jactorch.set_index_one_hot_(new_kernel, 1, indices, flatten_kernel)
    new_kernel = new_kernel[torch.nonzero(flatten_kernel)]
    return new_kernel.view(new_kernel.size(0), 1, kernel.size(0), kernel.size(1))


class MorphologyOp(nn.Module):
    def __init__(self, kernel_size, shape='rect'):
        super().__init__()
        kernel = get_morphology_kernel(shape, kernel_size)
        kernel = _cvt_morphology_kernel(kernel)
        self.register_buffer('kernel', kernel)

    def forward_morphology(self, image, op=None):
        k = self.kernel
        if image.dim() == 2:
            expanded = F.conv2d(image.unsqueeze(0).unsqueeze(0), k, padding=(k.shape[2] // 2, k.shape[3] // 2))
        elif image.dim() == 3:
            expanded = F.conv2d(image.unsqueeze(1), k, padding=(k.shape[2] // 2, k.shape[3] // 2))
        elif image.dim() == 4:
            assert image.size(1) == 1, 'Morphology operations support only gray-scale images.'
            expanded = F.conv2d(image, k, padding=(k.shape[2] // 2, k.shape[3] // 2))

        result = op(expanded, dim=1, keepdim=True)
        if type(result) is tuple:  # handle torch.min and torch.max
            result = result[0]

        if image.dim() == 2:
            return result[0, 0]
        elif image.dim() == 3:
            return result[:, 0]
        elif image.dim() == 4:
            return result

    def erode(self, image):
        return self.forward_morphology(image, torch.min)

    def dilate(self, image):
        return self.forward_morphology(image, torch.max)

    def open(self, image):
        return self.dilate(self.erode(image))

    def close(self, image):
        return self.erode(self.dilate(image))


class Erosion(MorphologyOp):
    def forward(self, image):
        return self.erode(image)


class Dilation(MorphologyOp):
    def forward(self, image):
        return self.dilate(image)


class Opening(MorphologyOp):
    def forward(self, image):
        return self.open(image)


class Closing(MorphologyOp):
    def forward(self, image):
        return self.close(image)


class MorphologicalGradient(MorphologyOp):
    def forward(self, image):
        return self.dilate(image) - self.erode(image)


class TopHat(MorphologyOp):
    def forward(self, image):
        return image - self.open(image)


class BlackHat(MorphologyOp):
    def forward(self, image):
        return self.close(image) - image


def erode(image, kernel_size, shape='rect'):
    return Erosion(kernel_size, shape).to(image.device)(image)


def dilate(image, kernel_size, shape='rect'):
    return Dilation(kernel_size, shape).to(image.device)(image)


def open(image, kernel_size, shape='rect'):
    return Opening(kernel_size, shape).to(image.device)(image)


def close(image, kernel_size, shape='rect'):
    return Closing(kernel_size, shape).to(image.device)(image)


def morph_grad(image, kernel_size, shape='rect'):
    return MorphologicalGradient(kernel_size, shape).to(image.device)(image)


def top_hat(image, kernel_size, shape='rect'):
    return TopHat(kernel_size, shape).to(image.device)(image)


def black_hat(image, kernel_size, shape='rect'):
    return BlackHat(kernel_size, shape).to(image.device)(image)

