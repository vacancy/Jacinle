#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : linalg.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.nn.functional as F

from .shape import concat_shape

__all__ = ['normalize', 'logaddexp', 'logsumexp', 'logmatmulexp']


def normalize(a, p=2, dim=-1, eps=1e-8):
    return a / a.norm(p, dim=dim, keepdim=True).clamp(min=eps)


def logaddexp(x, y):
    return torch.max(x, y) + torch.log(1 + torch.exp(-torch.abs(y - x)))


def logsumexp(inputs, dim=-1, keepdim=False):
    return (inputs - F.log_softmax(inputs, dim=dim)).mean(dim, keepdim=keepdim)


def logmatmulexp(mat1, mat2):
    mat1_shape = mat1.size()
    mat2_shape = mat2.size()
    mat1 = mat1.contiguous().view(-1, 1, mat1_shape[-1])
    mat2 = mat2.contiguous().view(1, mat2_shape[0], -1).transpose(1, 2)
    logprod = mat1 + mat2
    logsum = logsumexp(logprod, dim=-1)
    logsum.view(concat_shape(mat1_shape[:-1], mat2_shape[1:]))
    return logsum

