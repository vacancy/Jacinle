#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : coor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/16/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np
from ._utils import apply_affine_transform


def normalize_coor(img, coor):
    """
    Normalize image

    Args:
        img: (array): write your description
        coor: (array): write your description
    """
    coor = coor.copy()
    coor[:, 0] /= img.width
    coor[:, 1] /= img.height
    return coor


def denormalize_coor(img, coor):
    """
    Denormalize the image.

    Args:
        img: (array): write your description
        coor: (array): write your description
    """
    coor = coor.copy()
    coor[:, 0] *= img.width
    coor[:, 1] *= img.height
    return coor


def refresh_valid(img, coor):
    """
    Refresh the pixels of an image.

    Args:
        img: (array): write your description
        coor: (array): write your description
    """
    assert coor.shape[1] in (2, 3), 'Support only (x, y, valid) or (x, y) typed coordinates'
    has_valid_bit = coor.shape[1] == 3

    x, y = coor[:, 0], coor[:, 1]
    if has_valid_bit:
        v = coor[:, 2]
    else:
        v = 1

    valid = (v != 0) & (x >= 0) & (x < img.width) & (y >= 0) & (y < img.height)

    if has_valid_bit:
        out = coor.copy()
        invalid_indices = np.where(~valid)[0]
        out[invalid_indices, :] = 0
        return out
    else:
        valid_indices = np.where(valid)[0]
        return coor[valid_indices]


def crop(coor, x, y, w, h):
    """
    Crop a pixel or y and y.

    Args:
        coor: (array): write your description
        x: (array): write your description
        y: (array): write your description
        w: (array): write your description
        h: (array): write your description
    """
    coor = coor.copy()

    coor[:, 0] = (coor[:, 0] - x)
    coor[:, 1] = (coor[:, 1] - y)
    return coor


def pad(coor, padding):
    """
    Pad padding with padding.

    Args:
        coor: (array): write your description
        padding: (float): write your description
    """
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    else:
        assert len(padding) == 4

    coor = coor.copy()

    coor[:, 0] = (coor[:, 0] + padding[0])
    coor[:, 1] = (coor[:, 1] + padding[1])
    return coor


def hflip(img, coor):
    """
    Flip the image.

    Args:
        img: (array): write your description
        coor: (array): write your description
    """
    coor = coor.copy()
    coor[:, 0] = img.width - coor[:, 0]
    return coor


def vflip(img, coor):
    """
    Flip an image.

    Args:
        img: (array): write your description
        coor: (array): write your description
    """
    coor = coor.copy()
    coor[:, 1] = img.height - coor[:, 1]
    return coor


def resize(img, coor, size):
    """
    Resize image

    Args:
        img: (array): write your description
        coor: (array): write your description
        size: (int): write your description
    """
    h, w = size
    coor = coor.copy()
    coor[:, 0] = coor[:, 0] / img.width * w
    coor[:, 1] = coor[:, 1] / img.height * h
    return coor


def affine(coor, matrix):
    """
    Return the affine matrix.

    Args:
        coor: (array): write your description
        matrix: (array): write your description
    """
    coor = coor.copy()
    for i in range(coor.shape[0]):
        coor[i, :2] = apply_affine_transform(*coor[i, :2], matrix)

    return coor

