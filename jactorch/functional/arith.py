#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : arith.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/31/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Arithmetic operations."""

from typing import Any, Dict
import torch
import torch.nn.functional as F

__all__ = ['atanh', 'logit', 'log_sigmoid', 'tstat', 'soft_amax', 'soft_amin']


def atanh(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""
    Computes :math:`\mathrm{arc}\tanh(x)`.

    Args:
        x: input.
        eps: eps for numerical stability.

    Returns:
        :math:`\mathrm{arc}\tanh(x)`.

    """
    inner = (1 + x) / (1 - x).clamp(min=eps)
    return 0.5 * torch.log(inner.clamp(min=eps))


def logit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""
    Computes :math:`\mathrm{logit}(x)`.

    Args:
        x: input.
        eps: eps for numerical stability.

    Returns:
        :math:`\mathrm{logit}(x)`.

    """
    return -torch.log((1 / x.clamp(min=eps) - 1).clamp(min=eps))


def log_sigmoid(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes :math:`\log \sigma(x)`.

    Args:
        x: input.

    Returns:
        :math:`\log \sigma(x)`.
    """
    return -F.softplus(-x)


def tstat(x: torch.Tensor) -> Dict[str, Any]:
    """Tensor stats: produces a summary of the tensor, including shape, min, max, mean, and std.

    Args:
        x: input tensor.

    Returns:
        a dict of stats.
    """
    return {'shape': x.shape, 'min': x.min().item(), 'max': x.max().item(), 'mean': x.mean().item(), 'std': x.std().item()}


def soft_amax(x: torch.Tensor, dim: int, tau: float = 1.0, keepdim: bool = False) -> torch.Tensor:
    """Compute a soft maximum over the given dimension. It can be viewed as a differentiable version of :func:`torch.amax`.

    Args:
        x: input tensor.
        dim: dimension to compute the soft maximum.
        tau: temperature.
        keepdim: whether to keep the dimension.

    Returns:
        the soft maximum.
    """
    index = F.softmax(x / tau, dim=dim)
    return (x * index).sum(dim=dim, keepdim=keepdim)


def soft_amin(x, dim, tau=1.0, keepdim=False):
    """Compute a soft minimum over the given dimension. It can be viewed as a differentiable version of :func:`torch.amin`.

    Args:
        x: input tensor.
        dim: dimension to compute the soft minimum.
        tau: temperature.
        keepdim: whether to keep the dimension.

    Returns:
        the soft minimum.

    See also:

        :func:`soft_amax`

    """
    return -soft_amax(-x, dim=dim, tau=tau, keepdim=keepdim)

