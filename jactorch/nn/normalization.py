#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : normalization.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/10/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from jacinle.utils.numeric import prod

__all__ = ['LayerNorm']


def _unsqueeze_ft(tensor):
    """
    Return the tensor of the tensor.

    Args:
        tensor: (todo): write your description
    """
    return tensor.unsqueeze(0).unsqueeze(-1)


class LayerNorm(nn.Module):
    def __init__(self, num_features, dim=-1, eps=1e-5, affine=True):
        """
        Initialize the gradient.

        Args:
            self: (todo): write your description
            num_features: (int): write your description
            dim: (int): write your description
            eps: (float): write your description
            affine: (array): write your description
        """
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.dim = dim
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def reset_parameters(self):
        """
        Reset the parameters.

        Args:
            self: (todo): write your description
        """
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        dim = self.dim
        if dim < 0:
            dim += input.dim()
        input_shape = input.size()
        imm_shape = (prod(input_shape[:dim]), input_shape[dim], prod(input_shape[dim+1:]))

        input = input.view(imm_shape)
        mean = input.mean(1, keepdim=True)
        std = input.std(1, keepdim=True, unbiased=False)

        # Compute the output.
        if self.affine:
            output = (input - mean) * (_unsqueeze_ft(self.weight) / std) + _unsqueeze_ft(self.bias)
        else:
            output = (input - mean) / std

        return output.view(input_shape)
