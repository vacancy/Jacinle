#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : arith.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/31/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn.functional as F

__all__ = ['atanh', 'logit', 'log_sigmoid']


def atanh(x, eps=1e-8):
    """
    Computes :math:`arctanh(x)`.

    Args:
        x (FloatTensor): input.
        eps (float): eps for numerical stability.

    Returns:
        FloatTensor: :math:`arctanh(x)`.

    """
    return 0.5 * torch.log(( (1 + x) / (1 - x).clamp(min=eps) ).clamp(min=eps))


def logit(x, eps=1e-8):
    """
    Computes :math:`logit(x)`.

    Args:
        x (FloatTensor): input.
        eps (float): eps for numerical stability.

    Returns:
        FloatTensor: :math:`logit(x)`.

    """
    return -torch.log((1 / x.clamp(min=eps) - 1).clamp(min=eps))


def log_sigmoid(x):
    return -F.softplus(-x)

