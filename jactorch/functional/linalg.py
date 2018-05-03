# -*- coding: utf-8 -*-
# File   : linalg.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/02/2018
# 
# This file is part of Jacinle.

import torch.nn.functional as F

__all__ = ['normalize', 'logsumexp']


def normalize(a, p=2, dim=-1, eps=1e-8):
    return a / a.norm(p, dim=dim, keepdim=True).clamp(min=eps)


def logsumexp(inputs, dim=None, keepdim=False):
    return (inputs - F.log_softmax(inputs)).mean(dim, keepdim=keepdim)