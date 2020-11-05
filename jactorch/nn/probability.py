#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : probability.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from jactorch.functional.probability import normalize_prob

__all__ = ['ProbabilityLinear', 'ProbabilityBilinear', 'ProbabilityNLinear']


class ProbabilityLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, norm=True):
        """
        Initialize features.

        Args:
            self: (todo): write your description
            in_features: (int): write your description
            out_features: (int): write your description
            bias: (float): write your description
            norm: (todo): write your description
        """
        assert bias is False, 'Bias regularization for SOFTMAX is not implemented.'
        super().__init__(in_features, out_features, bias)
        self.norm = norm

    def forward(self, input):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        weight = self._regulize_parameter(self.weight)
        output = F.linear(input, weight, None)
        if self.norm:
            output = normalize_prob(output)
        return output

    def _regulize_parameter(self, p):
        """
        Regulize the parameter p.

        Args:
            self: (todo): write your description
            p: (todo): write your description
        """
        return F.softmax(p, dim=0)


class ProbabilityBilinear(nn.Bilinear):
    def __init__(self, in1_features, in2_features, out_features, bias=False, norm=True):
        """
        Initialize features.

        Args:
            self: (todo): write your description
            in1_features: (int): write your description
            in2_features: (int): write your description
            out_features: (int): write your description
            bias: (float): write your description
            norm: (todo): write your description
        """
        assert bias is False, 'Bias regularization for SOFTMAX is not implemented.'
        super().__init__(in1_features, in2_features, out_features, bias)
        self.norm = norm

    def forward(self, input1, input2):
        """
        Parameters ---------- input1 : int input2

        Args:
            self: (todo): write your description
            input1: (todo): write your description
            input2: (todo): write your description
        """
        weight = self._regulize_parameter(self.weight)
        output = F.bilinear(input1, input2, weight, None)
        if self.norm:
            output = normalize_prob(output)
        return output

    def _regulize_parameter(self, p):
        """
        Regulize the parameter p.

        Args:
            self: (todo): write your description
            p: (todo): write your description
        """
        return F.softmax(p, dim=0)


class ProbabilityNLinear(nn.Module):
    def __new__(cls, *nr_categories, bias=False, norm=True):
        """
        Create a new : class.

        Args:
            cls: (todo): write your description
            nr_categories: (todo): write your description
            bias: (float): write your description
            norm: (todo): write your description
        """
        if len(nr_categories) == 2:
            return ProbabilityLinear(*nr_categories, bias=bias, norm=norm)
        elif len(nr_categories) == 3:
            return ProbabilityBilinear(*nr_categories, bias=bias, norm=norm)
        else:
            return super().__new__(cls)

    def __init__(self, *nr_categories, bias=False, norm=True):
        """
        Initialize the internal state.

        Args:
            self: (todo): write your description
            nr_categories: (str): write your description
            bias: (float): write your description
            norm: (todo): write your description
        """
        super().__init__()
        assert bias is False, 'Bias regularization for SOFTMAX is not implemented.'

        self.nr_categories = nr_categories
        self.weight = Parameter(torch.Tensor(nr_categories[-1], *nr_categories[:-1]))
        self.reset_parameters()

        self.norm = norm

    def reset_parameters(self):
        """
        Reset the parameters.

        Args:
            self: (todo): write your description
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, *inputs):
        """
        Calculate the weights.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        f = self._regulize_parameter(self.weight)
        for i in reversed(inputs):
            f = (f * i).sum(dim=-1)
        if self.norm:
            f = normalize_prob(f)
        return f

    def _regulize_parameter(self, p):
        """
        Regulize the parameter p.

        Args:
            self: (todo): write your description
            p: (todo): write your description
        """
        return F.softmax(p, dim=0)
