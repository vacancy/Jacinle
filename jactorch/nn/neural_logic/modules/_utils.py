#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : _utils.py
# Author : Jiayuan Mao, Honghua Dong
# Email  : maojiayuan@gmail.com, dhh19951@gmail.com
# Date   : 03/28/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch

from jactorch.functional import meshgrid, meshgrid_exclude_self

__all__ = ['meshgrid', 'meshgrid_exclude_self', 'exclude_mask', 'mask_value']


def exclude_mask(input, cnt=2, dim=1):
    """
    Produce exclude mask. Specifically, for cnt=2, given an array a[i, j] of n * n, it produces
    a mask with size n * n where only a[i, j] = 1 if and only if (i != j).

    The operation is performed over [dim, dim + cnt) axes.
    """
    assert cnt > 0
    if dim < 0:
        dim += input.dim()
    n = input.size(dim)
    for i in range(1, cnt):
        assert n == input.size(dim + i)

    rng = torch.arange(0, n, dtype=torch.long, device=input.device)
    q = []
    for i in range(cnt):
        p = rng
        for j in range(cnt):
            if i != j:
                p = p.unsqueeze(j)
        p = p.expand((n,) * cnt)
        q.append(p)
    mask = q[0] == q[0]
    for i in range(cnt):
        for j in range(cnt):
            if i != j:
                mask *= q[i] != q[j]
    for i in range(dim):
        mask.unsqueeze_(0)
    for j in range(input.dim() - dim - cnt):
        mask.unsqueeze_(-1)

    return mask.float()


def mask_value(input, mask, value):
    assert input.dim() == mask.dim()
    return input * mask + value * (1 - mask)

