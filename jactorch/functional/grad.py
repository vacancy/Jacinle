# -*- coding: utf-8 -*-
# File   : grad.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 24/01/2018
# 
# This file is part of Jacinle.

from torch.autograd import Function

__all__ = ['grad_multi', 'zero_grad']


class GradMulti(Function):
    @staticmethod
    def forward(ctx, input, grad_multi):
        ctx.grad_multi = grad_multi
        output = input.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.grad_multi, None


grad_multi = GradMulti.apply


class ZeroGradV1(Function):
    @staticmethod
    def forward(ctx, input):
        output = input.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None


zero_grad_v1 = ZeroGradV1.apply


def zero_grad_v2(v):
    """Zero-grad the variable."""
    return v.detach()


zero_grad = zero_grad_v2
