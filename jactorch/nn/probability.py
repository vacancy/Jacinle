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
        assert bias is False, 'Bias regularization for SOFTMAX is not implemented.'
        super().__init__(in_features, out_features, bias)
        self.norm = norm

    def forward(self, input):
        weight = self._regulize_parameter(self.weight)
        output = F.linear(input, weight, None)
        if self.norm:
            output = normalize_prob(output)
        return output

    def _regulize_parameter(self, p):
        return F.softmax(p, dim=0)


class ProbabilityBilinear(nn.Bilinear):
    def __init__(self, in1_features, in2_features, out_features, bias=False, norm=True):
        assert bias is False, 'Bias regularization for SOFTMAX is not implemented.'
        super().__init__(in1_features, in2_features, out_features, bias)
        self.norm = norm

    def forward(self, input1, input2):
        weight = self._regulize_parameter(self.weight)
        output = F.bilinear(input1, input2, weight, None)
        if self.norm:
            output = normalize_prob(output)
        return output

    def _regulize_parameter(self, p):
        return F.softmax(p, dim=0)


class ProbabilityNLinear(nn.Module):
    def __new__(cls, *nr_categories, bias=False, norm=True):
        if len(nr_categories) == 2:
            return ProbabilityLinear(*nr_categories, bias=bias, norm=norm)
        elif len(nr_categories) == 3:
            return ProbabilityBilinear(*nr_categories, bias=bias, norm=norm)
        else:
            return super().__new__(cls)

    def __init__(self, *nr_categories, bias=False, norm=True):
        super().__init__()
        assert bias is False, 'Bias regularization for SOFTMAX is not implemented.'

        self.nr_categories = nr_categories
        self.weight = Parameter(torch.Tensor(nr_categories[-1], *nr_categories[:-1]))
        self.reset_parameters()

        self.norm = norm

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, *inputs):
        f = self._regulize_parameter(self.weight)
        for i in reversed(inputs):
            f = (f * i).sum(dim=-1)
        if self.norm:
            f = normalize_prob(f)
        return f

    def _regulize_parameter(self, p):
        return F.softmax(p, dim=0)
