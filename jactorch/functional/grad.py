#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : grad.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from torch.autograd import Function

__all__ = ['grad_multi', 'zero_grad']


class GradMulti(Function):
    @staticmethod
    def forward(ctx, input, grad_multi):
        """
        Parameters ----------

        Args:
            ctx: (todo): write your description
            input: (todo): write your description
            grad_multi: (bool): write your description
        """
        ctx.grad_multi = grad_multi
        output = input.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute backward backward pass

        Args:
            ctx: (todo): write your description
            grad_output: (bool): write your description
        """
        return grad_output * ctx.grad_multi, None


def grad_multi(input, grad_multi):
    """
    Scale the gradient with respect to the input.

    Args:
        input (Tensor): the input tensor.
        grad_multi (float): the constant for scaling up the gradient.

    Returns
        Tensor: of the same value as the input. But during the back-propagation,
        it will scale the gradient by `grad_multi`.

    """
    return GradMulti.apply(input, grad_multi)


class ZeroGradV1(Function):
    @staticmethod
    def forward(ctx, input):
        """
        R forward forward

        Args:
            ctx: (todo): write your description
            input: (todo): write your description
        """
        output = input.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward backward backward backward backward.

        Args:
            ctx: (todo): write your description
            grad_output: (bool): write your description
        """
        return None


def zero_grad_v1(input):
    """Zero-grad the variable."""
    return ZeroGradV1.apply(input)


def zero_grad_v2(v):
    """Zero-grad the variable."""
    return v.detach()


zero_grad = zero_grad_v2
