#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : meshgrid.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/31/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
from .shape import broadcast, concat_shape

__all__ = ['meshgrid', 'meshgrid_exclude_self']


def meshgrid(input1, input2=None, dim=-1):
    """Perform np.meshgrid along given axis. It will generate a new dimension after dim."""
    if input2 is None:
        input2 = input1
    if dim < 0:
        dim += input1.dim()
    n, m = input1.size(dim), input2.size(dim)
    x = broadcast(input1.unsqueeze(dim + 1), dim + 1, m)
    y = broadcast(input2.unsqueeze(dim + 0), dim + 0, n)
    return x, y


def meshgrid_exclude_self(input, dim=1):
    """
    Exclude self from the grid. Specifically, given an array a[i, j] of n * n, it produces
    a new array with size n * (n - 1) where only a[i, j] (i != j) is preserved.

    The operation is performed over dim and dim +1 axes.
    """
    if dim < 0:
        dim += input.dim()

    n = input.size(dim)
    assert n == input.size(dim + 1)

    # exclude self-attention
    rng = torch.arange(0, n, dtype=torch.long, device=input.device)
    rng_n1 = rng.unsqueeze(1).expand((n, n))
    rng_1n = rng.unsqueeze(0).expand((n, n))
    mask_self = (rng_n1 != rng_1n)

    for i in range(dim):
        mask_self.unsqueeze_(0)
    for j in range(input.dim() - dim - 2):
        mask_self.unsqueeze_(-1)
    target_shape = concat_shape(input.size()[:dim], n, n-1, input.size()[dim+2:])

    return input.masked_select(mask_self).view(target_shape)
