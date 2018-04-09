# -*- coding: utf-8 -*-
# File   : sample.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/04/2018
#
# This file is part of Jacinle.

import torch
import torch.autograd as autograd

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
        input_flatten = input.view(-1, input.size(-1))
        rand = torch.multinomial(input_flatten, 1).view(input.size()[:-1], 1)
        rand.transpose_(dim, -1).unsqueeze_(-1)
        return rand

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def sample_multinomial(input, dim=-1):
    return SampleMultinomial.apply(input, dim)
