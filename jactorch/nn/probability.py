# -*- coding: utf-8 -*-
# File   : probability.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/02/2018
# 
# This file is part of Jacinle.


import torch.nn as nn
import torch.nn.functional as F

from jactorch.functional.probability import normalize_prob


class ProbabilityBilinear(nn.Bilinear):
    def __init__(self, in1_features, in2_features, out_features, bias=False, norm=True):
        super().__init__(in1_features, in2_features, out_features, bias)
        self.norm = norm
        assert bias is False, 'Bias regularization for SOFTMAX is not implemented.'

    def forward(self, input1, input2):
        weight = self._regulize_parameter(self.weight)
        bias = self._regulize_parameter(self.bias)
        output = F.bilinear(input1, input2, weight, bias)
        if self.norm:
            output = normalize_prob(output)
        return output

    def _regulize_parameter(self, p):
        if p is None:
            return None
        return F.softmax(p, dim=0)
