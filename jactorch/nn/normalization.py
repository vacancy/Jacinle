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
    return tensor.unsqueeze(0).unsqueeze(-1)


class LayerNorm(nn.Module):
    def __init__(self, num_features, dim=-1, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.dim = dim
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input):
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
