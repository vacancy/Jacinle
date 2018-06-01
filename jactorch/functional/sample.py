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

from .indexing import one_hot_nd

__all__ = ['sample_bernoulli', 'sample_multinomial']


class SampleBernoulli(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        rand = input.new(*input.size())
        torch.rand(input.size(), out=rand)
        return (rand > input).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

sample_bernoulli = SampleBernoulli.apply


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
