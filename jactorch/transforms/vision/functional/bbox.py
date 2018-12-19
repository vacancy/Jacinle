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
    bbox = bbox.copy()
    bbox[:, 0] /= img.width
    bbox[:, 1] /= img.height
    bbox[:, 2] /= img.width
    bbox[:, 3] /= img.height
    return bbox


def denormalize_box(img, bbox):
    bbox = bbox.copy()
    bbox[:, 0] *= img.width
    bbox[:, 1] *= img.height
    bbox[:, 2] /= img.width
    bbox[:, 3] /= img.height
    return bbox


def crop(bbox, x, y, w, h):
    bbox = bbox.copy()

    bbox[:, 0] = (bbox[:, 0] - x)
    bbox[:, 1] = (bbox[:, 1] - w)
    return bbox


def pad(bbox, padding):
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
    bbox = bbox.copy()
    bbox[:, 0] = img.width - bbox[:, 0] - bbox[:, 2]
    return bbox


def vflip(img, bbox):
    bbox = bbox.copy()
    bbox[:, 1] = img.height - bbox[:, 1] - bbox[:, 3]
    return bbox


def resize(img, bbox, size):
    w, h = size
    bbox = bbox.copy()
    bbox[:, 0] = bbox[:, 0] / img.width * w
    bbox[:, 1] = bbox[:, 1] / img.height * h
    bbox[:, 2] = bbox[:, 2] / img.width * w
    bbox[:, 3] = bbox[:, 3] / img.height * h
    return bbox


def affine(bbox, matrix):
    bbox = bbox.copy()
    for i in range(bbox.shape[0]):
        bbox[i, :2] = apply_affine_transform(*bbox[i, :2], matrix)
        raise NotImplementedError()

    return bbox

