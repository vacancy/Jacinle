#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : quantization.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/09/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Basic quantization functions with a straight-through gradient estimator."""

__all__ = ['quantize', 'randomized_quantize']

import torch
import torch.autograd as ag


class _Quantize(ag.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _RandomizedQuantize(ag.Function):
    @staticmethod
    def forward(ctx, x):
        rand = torch.rand(x.size())
        return (rand > x).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize(x: torch.Tensor) -> torch.Tensor:
    """Quantize a tensor to binary values: ``(x > 0.5).float()``. This function implements the straight-through gradient estimator.

    Args:
        x: the input tensor.

    Returns:
        the quantized tensor.
    """
    return _Quantize.apply(x)


def randomized_quantize(x: torch.Tensor) -> torch.Tensor:
    """Quantize a tensor to binary values: ``(rand() > x).float()``. This function implements the straight-through gradient estimator.

    Args:
        x: the input tensor.

    Returns:
        the quantized tensor.
    """
    return _RandomizedQuantize.apply(x)

