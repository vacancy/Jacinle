#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mask.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/08/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['masked_average']


def masked_average(tensor, mask, eps=1e-8):
    tensor = tensor.float()
    mask = mask.float()
    masked = tensor * mask
    return masked.sum() / mask.sum().clamp(min=eps)
