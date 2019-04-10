#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : conv.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/04/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from jacinle.utils.argument import get_2dshape
from jactorch.nn.cnn.functional import ConvBorderMode, padding_nd

__all__ = [
    'MaxPoolingKernelDef', 'custom_kernel', 'CustomKernel'
]


class MaxPoolingKernelDef(object):
    def __init__(self, kernel_size):
        self.kernel_size = get_2dshape(kernel_size)


def custom_kernel(image, k, border_mode='zero'):
    border_mode = ConvBorderMode.from_string(border_mode)

    if not isinstance(k, MaxPoolingKernelDef):
        if not torch.is_tensor(k):
            k = torch.tensor(k, device=image.device, dtype=torch.float32)

        if k.dim() == 2:
            k = k.unsqueeze(0).unsqueeze(0)
        elif k.dim() == 3:
            k = k.unsqueeze(1)
        elif k.dim() == 4:
            pass
        else:
            raise ValueError('Unsupported kernel size: {}.'.format(k.size()))

        assert k.size(2) % 2 == 1 and k.size(3) % 2 == 1
        kernel_shape = k.shape[2:4]
        padding = (k.shape[2] // 2, k.shape[3] // 2)
    else:
        kernel_shape = k.kernel_size
        assert kernel_shape[0] % 2 == 1 and kernel_shape[1] % 2 == 1
        padding = (kernel_shape[0] // 2, kernel_shape[1] // 2)

    image_dim = image.dim()
    image_size = image.size()
    if image_dim not in (2, 3, 4):
        raise ValueError('Unsupported image dim: {}.'.format(image_dim))
    for i in range(4 - image_dim):
        image = image.unsqueeze(0)
    image, extra_padding = padding_nd(image, kernel_shape, None, 'same', border_mode)

    if not isinstance(k, MaxPoolingKernelDef):
        return F.conv2d(
            image.contiguous().view((image.shape[0] * image.shape[1], 1) + image.shape[2:]),
            k,
            padding=extra_padding
        ).view(image_size)
    else:
        return F.max_pool2d(image, kernel_shape, stride=1, padding=extra_padding).view(image_size)



class CustomKernel(nn.Module):
    def __init__(self, kernel, border_mode='zero'):
        super().__init__()
        if not isinstance(kernel, MaxPoolingKernelDef):
            if not torch.is_tensor(kernel):
                kernel = torch.tensor(kernel, dtype=torch.float32)
            self.register_buffer('kernel', kernel)
        else:
            self.kernel = kernel

        self.border_mode = ConvBorderMode.from_string(border_mode)

    def forward(self, input):
        return custom_kernel(input, self.kernel, self.border_mode)

