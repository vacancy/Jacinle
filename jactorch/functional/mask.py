# -*- coding: utf-8 -*-
# File   : mask.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/03/2018
# 
# This file is part of Jacinle.

__all__ = ['masked_average']


def masked_average(tensor, mask, eps=1e-8):
    tensor = tensor.float()
    mask = mask.float()
    masked = tensor * mask
    return masked.sum() / mask.sum().clamp(min=eps)
