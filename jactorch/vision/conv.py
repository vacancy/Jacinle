#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : conv.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com # Date   : 01/04/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'custom_kernel', 'CustomKernel',
]

def custom_kernel(image, k):
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

    image_dim = image.dim()
    if image_dim == 2:
        return F.conv2d(
            image.unsqueeze(0).unsqueeze(0),
            k,
            padding=(k.shape[2] // 2, k.shape[3] // 2)
        )[0, 0]
    elif image_dim == 3:
        return F.conv2d(
            image.unsqueeze(1),
            k,
            padding=(k.shape[2] // 2, k.shape[3] // 2)
        )[:, 0]
    elif image_dim == 4:
        image_size = image.size()
        return F.conv2d(
            image.contiguous().view((image_size[0] * image_size[1], 1) + image_size[2:]),
            k,
            padding=(k.shape[2] // 2, k.shape[3] // 2)
        ).view(image_size)
    else:
        raise ValueError('Unsupported image dim: {}.'.format(image_dim))


class CustomKernel(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        if not torch.is_tensor(kernel):
            kernel = torch.tensor(kernel, dtype=torch.float32)
        self.register_buffer('kernel', kernel)

    def forward(self, input):
        return custom_kernel(input, self.kernel)

