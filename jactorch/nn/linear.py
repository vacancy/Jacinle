#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : linear.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/10/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.nn as nn

from jactorch.functional import concat_shape

__all__ = ['AnyDimLinear']


class AnyDimLinear(nn.Linear):
    def __init__(self, in_features, out_features, hidden_dim=-1, bias=True):
        """
        Initialize features.

        Args:
            self: (todo): write your description
            in_features: (int): write your description
            out_features: (int): write your description
            hidden_dim: (int): write your description
            bias: (float): write your description
        """
        super().__init__(in_features, out_features, bias=bias)
        self.hidden_dim = hidden_dim

    def forward(self, input):
        """
        R forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        input = input.transpose(self.hidden_dim, -1)
        other_dims = input.size()[:-1]
        input = input.contiguous().view(-1, self.in_features)
        output = super().forward(input)
        output = output.view(concat_shape(other_dims, -1))
        output = output.transpose(self.hidden_dim, -1)
        return output
