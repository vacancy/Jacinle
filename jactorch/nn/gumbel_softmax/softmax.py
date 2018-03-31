# -*- coding: utf-8 -*-
# File   : softmax.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/04/2018
# 
# This file is part of Jacinle.

import torch.nn as nn

from .functional import gumbel_softmax


class GumbelSoftmax(nn.Module):
    def __init__(self, dim=-1, tau=1.0, hard=False, eps=1e-10):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.hard = hard
        self.eps = eps

    def forward(self, logits):
        return gumbel_softmax(logits, dim=self.dim, tau=self.tau, hard=self.hard, eps=self.eps)
