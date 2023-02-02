#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : indexing.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from typing import Union, Sequence

import numpy as np

from .meta import isndarray

__all__ = ['one_hot', 'one_hot_nd', 'index_select_batch']


def one_hot(label: np.ndarray, nr_classes: int, dtype='float32') -> np.ndarray:
    """Convert a label array to one-hot array. This function works for either 0d (scalar) or 1d (vector) label array.
    If you want to convert higher dimensional label array, use :meth:`one_hot_nd` instead.

    Args:
        label: the label array.
        nr_classes: the number of classes.
        dtype: the data type of the one-hot array.

    Returns:
        the one-hot array.
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
    """Convert a label array to one-hot array.

    Args:
        label: the label array.
        nr_classes: the number of classes.
        dtype: the data type of the one-hot array.

    Returns:
        the one-hot array.
    """
    shape = label.shape
    return one_hot(label.reshape(-1), nr_classes, dtype=dtype).reshape(shape + (nr_classes, ))


def index_select_batch(data: Union[np.ndarray, Sequence[np.ndarray]], indices: Union[Sequence[int], np.ndarray]) -> np.ndarray:
    """Gather ``indices`` as batch indices from ``data``, which can either be typical nd array or a list of nd array.

    Args:
        data: the data array.
        indices: the indices to be selected.

    Returns:
        the selected data.
    """
    assert isinstance(indices, (tuple, list)) or (isndarray(indices) and len(indices.shape) == 1)

    if isndarray(data):
        return data[indices]

    assert len(data) > 0 and len(indices) > 0

    sample = np.array(data[0])  # Try to convert the first element to a typical nd array.
    output = np.empty((len(indices), ) + sample.shape, dtype=sample.dtype)
    for i, j in enumerate(indices):
        output[i] = data[j]
    return output
