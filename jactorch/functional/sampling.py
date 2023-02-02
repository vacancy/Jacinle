#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : sample.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/09/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Sampling functions."""

import torch
import torch.autograd as autograd

from jacinle.utils.vendor import requires_vendors
from .indexing import one_hot_nd

__all__ = ['sample_bernoulli', 'sample_multinomial', 'choice']


class SampleBernoulli(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rand = x.new(*x.size())
        torch.rand(x.size(), out=rand)
        return (rand > x).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def sample_bernoulli(x):
    """Sample from a Bernoulli distribution.

    Args:
        x: the probability of the Bernoulli distribution.

    Returns:
        A tensor with the same shape as ``x``, where each element is sampled from the corresponding Bernoulli distribution.
    """
    return SampleBernoulli.apply(x)


class SampleMultinomial(autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        x = x.transpose(dim, -1)
        x_flatten = x.contiguous().view(-1, x.size(-1))
        rand = torch.multinomial(x_flatten, 1).view(x.size()[:-1])
        output = one_hot_nd(rand, x.size(dim))
        output = output.transpose(dim, -1)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def sample_multinomial(x, dim=-1):
    """Sample from a multinomial distribution.

    Args:
        x: the probability of the multinomial distribution.
        dim: the dimension of the categories.

    Returns:
        A tensor with the same shape as ``x``, where each element is sampled from the corresponding multinomial distribution.
    """
    return SampleMultinomial.apply(x, dim)


@requires_vendors('pytorch_reservoir')
def choice(a, k=1, replace=True, p=None, dtype=None, device=None):
    """
    Generates a random sample from a given 1-D array.

    Args:
        a (torch.Tensor): 1-D tensor or int.
        k (int, optional): number of samples to be drawn.
        replace (bool, optional): whether the sample is with or without replacement.
        p (torch.Tensor, optional): an optional weight parameter. Not necessarily to be normalized.

    Returns:
        torch.Tensor: 1-D outputs of k sampled data from `a`.
    """
    import pytorch_reservoir

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

