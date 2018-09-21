#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : kernel.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Useful utilities for kernel-based attention mechanism."""

import torch

from .linalg import normalize

__all__ = ['inverse_distance', 'cosine_distance', 'dot']


def inverse_distance(f_lookup, f, p=2, eps=1e-8):
    """
    Inverse distance kernel.

    Args:
        f_lookup (FloatTensor): features of the lookup keys
        f (FloatTensor): features of the value keys

    Returns:
        FloatTensor: the attention mask for each lookup keys.

    """

    n, m, k = f_lookup.size(0), f.size(0), f.size(1)
    f_lookup = f_lookup.view(n, 1, k).expand(n, m, k)
    f = f.view(1, m, k).expand(n, m, k)

    # TODO(Jiayuan Mao @ 05/26): this function can be optimized.
    dist = (f_lookup - f).norm(p, dim=2)
    return 1. / dist.clamp(min=eps)


def cosine_distance(f_lookup, f):
    """
    Cosine distance kernel.

    Args:
        f_lookup (FloatTensor): features of the lookup keys
        f (FloatTensor): features of the value keys

    Returns:
        FloatTensor: the attention mask for each lookup keys.

    """
    f_lookup = normalize(f_lookup, 2, dim=1)
    f = normalize(f, 2, dim=1)

    return torch.mm(f_lookup, f.t())


def dot(f_lookup, f):
    """
    Dot product kernel, essentially a cosine distance kernel without normalization.

    Args:
        f_lookup (FloatTensor): features of the lookup keys
        f (FloatTensor): features of the value keys

    Returns:
        FloatTensor: the attention mask for each lookup keys.

    """
    return torch.mm(f_lookup, f.t())

