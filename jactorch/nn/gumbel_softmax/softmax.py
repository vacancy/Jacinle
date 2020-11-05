#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : softmax.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/01/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.nn as nn

from .functional import gumbel_softmax, greedy_softmax, general_softmax, SoftmaxImplmentation

__all__ = ['SoftmaxImplmentation', 'GumbelSoftmax', 'GeneralSoftmax']


class GumbelSoftmax(nn.Module):
    def __init__(self, dim=-1, tau=1.0, hard=False, eps=1e-10):
        """
        Initialize the tau.

        Args:
            self: (todo): write your description
            dim: (int): write your description
            tau: (todo): write your description
            hard: (todo): write your description
            eps: (float): write your description
        """
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.hard = hard
        self.eps = eps

    def forward(self, logits, mask=None):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            logits: (todo): write your description
            mask: (todo): write your description
        """
        if self.training:
            return gumbel_softmax(logits, dim=self.dim, tau=self.tau, hard=self.hard, mask=mask, eps=self.eps)
        else:
            return greedy_softmax(logits, dim=self.dim, mask=mask)


class GeneralSoftmax(nn.Module):
    def __init__(self, dim=-1, tau=1.0, impl='standard'):
        """
        Initialize the gradient.

        Args:
            self: (todo): write your description
            dim: (int): write your description
            tau: (todo): write your description
            impl: (todo): write your description
        """
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.impl = SoftmaxImplmentation.from_string(impl)

    def forward(self, logits, mask=None):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            logits: (todo): write your description
            mask: (todo): write your description
        """
        return general_softmax(logits, dim=self.dim, tau=self.tau, impl=self.impl, mask=mask, training=self.training)

