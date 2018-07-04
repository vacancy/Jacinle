#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : arith.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/31/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['atanh', 'logit']

import torch


def atanh(x, eps=1e-8):
    return 0.5 * torch.log(( (1 + x) / (1 - x).clamp(min=eps) ).clamp(min=eps))


def logit(x, eps=1e-8):
    return -torch.log((1 / x.clamp(min=eps) - 1).clamp(min=eps))

