#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : quantization.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/09/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
Quantization functionals.
"""

__all__ = ['quantize', 'randomized_quantize']

import torch
import torch.autograd as ag


class Quantize(ag.Function):
    @staticmethod
    def forward(ctx, input):
        """
        Forward forward forward.

        Args:
            ctx: (todo): write your description
            input: (todo): write your description
        """
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward computation of a backward.

        Args:
            ctx: (todo): write your description
            grad_output: (bool): write your description
        """
        return grad_output


class RandomizedQuantize(ag.Function):
    @staticmethod
    def forward(ctx, input):
        """
        Forward computation.

        Args:
            ctx: (todo): write your description
            input: (todo): write your description
        """
        rand = torch.rand(input.size())
        return (rand > input).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward computation of a backward.

        Args:
            ctx: (todo): write your description
            grad_output: (bool): write your description
        """
        return grad_output


quantize = Quantize.apply
randomized_quantize = RandomizedQuantize.apply

