#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : random.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/11/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch

__all__ = ['choice']


def choice(a, k=1, replace=True, p=None, dtype=None, device=None):
    """
    Generates a random sample from a given 1-D array.

    Args:
        a (torch.Tensor): 1-D tensor or int.
        k (int, optional): number of samples to be drawn.
        replace (bool, optional): whether the sample is with or without replacement.
        p (torch.Tensor, optional): an optional weight parameter. Not necessarily to be normalized.

    Returns:
        sampled (torch.Tensor): 1-D outputs of k sampled data from `a`.
    """
    try:
        import pytorch_reservoir
    except ImportError as e:
        raise ImportError('Cannot load pytorch_reservoir. Make sure you have it as a vendor for Jacinle.') from e

    if isinstance(a, int):
        a = torch.arange(a, dtype=dtype, device=device)
    if not torch.is_tensor(a):
        a = torch.tensor(a)

    assert a.dim() == 1, 'jactorch.choice supports only 1-D input.'
    a = a.to(dtype=dtype, device=device)

    if p is not None:
        assert a.size() == p.size()
        return pytorch_reservoir.choice(a, p.to(device=device), replace, k)

    return pytorch_reservoir.choice(a, replace, k)

