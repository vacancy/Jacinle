#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : indexing.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np
from .nd import isndarray


def one_hot(label, nr_classes, dtype='float32'):
    """
    Convert a one - hot_classes.

    Args:
        label: (str): write your description
        nr_classes: (todo): write your description
        dtype: (todo): write your description
    """
    if isinstance(label, int) or (isndarray(label) and len(label.shape) == 0):
        out = np.zeros(nr_classes, dtype=dtype)
        out[int(label)] = 1
        return out

    assert len(label.shape) == 1
    nr_labels = label.shape[0]
    out = np.zeros((nr_labels, nr_classes), dtype=dtype)
    out[np.arange(nr_labels), label] = 1
    return out


def one_hot_nd(label, nr_classes, dtype='float32'):
    """
    Convert one - hot_nd_classes.

    Args:
        label: (array): write your description
        nr_classes: (todo): write your description
        dtype: (todo): write your description
    """
    shape = label.shape
    return one_hot(label.reshape(-1), nr_classes, dtype=dtype).reshape(shape + (nr_classes, ))


def index_select_batch(data, indices):
    """Gather `indices` as batch indices from `data`, which can either be typical nd array or a
    list of nd array"""
    assert isinstance(indices, (tuple, list)) or (isndarray(indices) and len(indices.shape) == 1)

    if isndarray(data):
        return data[indices]

    assert len(data) > 0 and len(indices) > 0

    sample = np.array(data[0])  # Try to convert the first element to a typical nd array.
    output = np.empty((len(indices), ) + sample.shape, dtype=sample.dtype)
    for i, j in enumerate(indices):
        output[i] = data[j]
    return output
