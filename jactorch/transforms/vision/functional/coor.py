#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : coor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/16/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from ._utils import apply_affine_transform


def normalize_coor(img, coor):
    coor = coor.copy()
    coor[:, 0] /= img.width
    coor[:, 1] /= img.height
    return coor


def denormalize_coor(img, coor):
    coor = coor.copy()
    coor[:, 0] *= img.width
    coor[:, 1] *= img.height
    return coor


def refresh_valid(img, coor):
    if coor.shape[1] == 2:
        return img, coor
    assert coor.shape[1] == 3, 'Support only (x, y, valid) or (x, y) typed coordinates'
    out = coor.copy()
    for i, (x, y, v) in enumerate(coor):
        valid = (v == 1) and (x >= 0) and (x < img.width) and (y >= 0) and (y < img.height)
        if not valid:
            out[i, :] = 0
    return out


def crop(coor, x, y, w, h):
    coor = coor.copy()

    coor[:, 0] = (coor[:, 0] - x)
    coor[:, 1] = (coor[:, 1] - y)
    return coor


def pad(coor, padding):
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
    coor = coor.copy()
    coor[:, 0] = img.width - coor[:, 0]
    return coor


def vflip(img, coor):
    coor = coor.copy()
    coor[:, 1] = img.height - coor[:, 1]
    return coor


def resize(img, coor, size):
    w, h = size
    coor = coor.copy()
    coor[:, 0] = coor[:, 0] / img.width * w
    coor[:, 1] = coor[:, 1] / img.height * h
    return coor


def affine(coor, matrix):
    coor = coor.copy()
    for i in range(coor.shape[0]):
        coor[i, :2] = apply_affine_transform(*coor[i, :2], matrix)

    return coor

