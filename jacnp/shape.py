#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : shape.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/03/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np

__all__ = ['unsqueeze', 'unsqueeze_as', 'softmax']


def unsqueeze(x, *axes):
    for axis in axes:
        x = np.expand_dims(x, axis)
    return x


def unsqueeze_as(x, y, *x_axes_in_y):
    for i in range(len(y.shape)):
        if i not in x_axes_in_y:
            x = unsqueeze(x, i)
    return x


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)
