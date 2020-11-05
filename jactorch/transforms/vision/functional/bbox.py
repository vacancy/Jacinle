#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : bbox.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/14/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from ._utils import apply_affine_transform


def normalize_bbox(img, bbox):
    """
    Normalize a bounding box.

    Args:
        img: (array): write your description
        bbox: (array): write your description
    """
    bbox = bbox.copy()
    bbox[:, 0] /= img.width
    bbox[:, 1] /= img.height
    bbox[:, 2] /= img.width
    bbox[:, 3] /= img.height
    return bbox


def denormalize_box(img, bbox):
    """
    Denormalize the image

    Args:
        img: (array): write your description
        bbox: (array): write your description
    """
    bbox = bbox.copy()
    bbox[:, 0] *= img.width
    bbox[:, 1] *= img.height
    bbox[:, 2] /= img.width
    bbox[:, 3] /= img.height
    return bbox


def refresh_valid(img, bbox):
    """
    Refresh the bounding boxes.

    Args:
        img: (array): write your description
        bbox: (array): write your description
    """
    assert bbox.shape[1] in (4, 5), 'Support only (x1, y1, x2, y2, valid) or (x1, y1, x2, y2) typed coordinates'
    has_valid_bit = bbox.shape[1] == 5

    out = bbox.copy()
    out[:, 0] = np.fmax(out[:, 0], 0)
    out[:, 1] = np.fmax(out[:, 1], 0)
    out[:, 2] = np.fmin(out[:, 2], img.width)
    out[:, 3] = np.fmin(out[:, 3], img.height)

    if has_valid_bit:
        v = bbox[:, 4]
    else:
        v = 1

    valid = (v != 0) & ((out[:, 2] - out[:, 0]) * (out[:, 3] - out[:, 1]) > 0)

    if has_valid_bit:
        invalid_indices = np.where(~valid)[0]
        out[invalid_indices, :] = 0
    else:
        valid_indices = np.where(valid)[0]
        return out[valid_indices]


def crop(bbox, x, y, w, h):
    """
    Crop a bounding box.

    Args:
        bbox: (array): write your description
        x: (array): write your description
        y: (array): write your description
        w: (array): write your description
        h: (array): write your description
    """
    bbox = bbox.copy()

    bbox[:, 0] = (bbox[:, 0] - x)
    bbox[:, 1] = (bbox[:, 1] - w)
    return bbox


def pad(bbox, padding):
    """
    Pad a bounding box.

    Args:
        bbox: (array): write your description
        padding: (float): write your description
    """
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    else:
        assert len(padding) == 4

    bbox = bbox.copy()

    bbox[:, 0] = (bbox[:, 0] + padding[0])
    bbox[:, 1] = (bbox[:, 1] + padding[1])
    return bbox


def hflip(img, bbox):
    """
    Flip a bounding box

    Args:
        img: (array): write your description
        bbox: (array): write your description
    """
    bbox = bbox.copy()
    bbox[:, 0] = img.width - bbox[:, 0] - bbox[:, 2]
    return bbox


def vflip(img, bbox):
    """
    Flip a bounding

    Args:
        img: (array): write your description
        bbox: (array): write your description
    """
    bbox = bbox.copy()
    bbox[:, 1] = img.height - bbox[:, 1] - bbox[:, 3]
    return bbox


def resize(img, bbox, size):
    """
    Resize the image

    Args:
        img: (array): write your description
        bbox: (array): write your description
        size: (int): write your description
    """
    h, w = size
    bbox = bbox.copy()
    bbox[:, 0] = bbox[:, 0] / img.width * w
    bbox[:, 1] = bbox[:, 1] / img.height * h
    bbox[:, 2] = bbox[:, 2] / img.width * w
    bbox[:, 3] = bbox[:, 3] / img.height * h
    return bbox


def affine(bbox, matrix):
    """
    Return a new bounding box.

    Args:
        bbox: (array): write your description
        matrix: (array): write your description
    """
    bbox = bbox.copy()
    for i in range(bbox.shape[0]):
        bbox[i, :2] = apply_affine_transform(*bbox[i, :2], matrix)
        raise NotImplementedError()

    return bbox

