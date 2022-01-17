#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : arith.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/31/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn.functional as F

__all__ = ['atanh', 'logit', 'log_sigmoid', 'tstat', 'soft_amax', 'soft_amin']


def atanh(x, eps=1e-8):
    """
    Computes :math:`arctanh(x)`.

    Args:
        x (FloatTensor): input.
        eps (float): eps for numerical stability.

    Returns:
        FloatTensor: :math:`arctanh(x)`.

    """
    return 0.5 * torch.log(( (1 + x) / (1 - x).clamp(min=eps) ).clamp(min=eps))


def logit(x, eps=1e-8):
    """
    Computes :math:`logit(x)`.

    Args:
        x (FloatTensor): input.
        eps (float): eps for numerical stability.

    Returns:
        FloatTensor: :math:`logit(x)`.

    """
    return -torch.log((1 / x.clamp(min=eps) - 1).clamp(min=eps))


def log_sigmoid(x):
    return -F.softplus(-x)


def tstat(x):
    """Tensor stats: produces a summary of the tensor."""
    return {'shape': x.shape, 'min': x.min().item(), 'max': x.max().item(), 'mean': x.mean().item(), 'std': x.std().item()}


def soft_amax(x, dim, tau=1.0, keepdim=False):
    """Return a soft version of x.max(dim=dim)."""
    index = F.softmax(x / tau, dim=dim)
    return (x * index).sum(dim=dim, keepdim=keepdim)


def soft_amin(x, dim, tau=1.0, keepdim=False):
    """See `soft_amax`."""
    return -soft_amax(-x, dim=dim, tau=tau, keepdim=keepdim)
