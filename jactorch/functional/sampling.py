#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : sample.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/09/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.autograd as autograd

from jacinle.utils.vendor import requires_vendors
from .indexing import one_hot_nd

__all__ = ['sample_bernoulli', 'sample_multinomial', 'choice']


class SampleBernoulli(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        rand = input.new(*input.size())
        torch.rand(input.size(), out=rand)
        return (rand > input).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def sample_bernoulli(input):
    return SampleBernoulli.apply(input)


class SampleMultinomial(autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        input = input.transpose(dim, -1)
        input_flatten = input.contiguous().view(-1, input.size(-1))
        rand = torch.multinomial(input_flatten, 1).view(input.size()[:-1])
        output = one_hot_nd(rand, input.size(dim))
        output = output.transpose(dim, -1)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def sample_multinomial(input, dim=-1):
    return SampleMultinomial.apply(input, dim)


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

