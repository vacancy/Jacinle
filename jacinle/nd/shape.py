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


def unsqueeze(x: np.ndarray, *axes: int) -> np.ndarray:
    """Unsqueeze the numpy array at the given axes. Similar to the PyTorch's :meth:`torch.unsqueeze`.

    Args:
        x: the input array.
        axes: the axes to unsqueeze.

    Returns:
        the unsqueezed array.
    """
    for axis in axes:
        x = np.expand_dims(x, axis)
    return x


def unsqueeze_as(x: np.ndarray, y: np.ndarray, *x_axes_in_y: int) -> np.ndarray:
    """Unsqueeze the numpy array ``x`` as the shape of ``y``. The corresponding axes in ``x`` are specified by ``x_axes_in_y``.

    Example:
        >>> x = np.zeros((2, 3, 4))
        >>> y = np.zeros((2, 3, 4, 5))
        >>> unsqueeze_as(x, y, 0, 1, 2).shape
        (2, 3, 4, 1)

    Args:
        x: the input array.
        y: the target array.
        x_axes_in_y: the axes in ``x`` that correspond to the axes in ``y``.

    Returns:
        the unsqueezed array.
    """
    for i in range(len(y.shape)):
        if i not in x_axes_in_y:
            x = unsqueeze(x, i)
    return x


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    r"""Compute the softmax of the input array at the given axis.

    .. math::

        \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}

    Args:
        x: the input array.
        axis: the axis to apply softmax.

    Returns:
        the softmax array.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)
