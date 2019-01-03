#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/03/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import math

from PIL import Image
import numpy as np

import torchvision.transforms.functional as TF
import jactorch.transforms.image.functional as jac_tf
from jacinle.utils.argument import get_2dshape


def normalize_coor(img, coor):
    coor = coor.copy()
    coor[:, 0] /= img.width
    coor[:, 1] /= img.height
    return img, coor


def denormalize_coor(img, coor):
    coor = coor.copy()
    coor[:, 0] *= img.width
    coor[:, 1] *= img.height
    return img, coor


def crop(img, coor, i, j, h, w):
    coor = coor.copy()

    coor[:, 0] = (coor[:, 0] - j / img.width) * (img.width / w)
    coor[:, 1] = (coor[:, 1] - i / img.height) * (img.height / h)
    return TF.crop(img, i, j, h, w), coor


def center_crop(img, coor, output_size):
    output_size = get_2dshape(output_size)
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, coor, i, j, th, tw)


def pad(img, coor, padding, mode='constant', fill=0):
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    else:
        assert len(padding) == 4

    img_new = jac_tf.pad(img, padding, mode=mode, fill=fill)
    coor = coor.copy()

    coor[:, 0] = (coor[:, 0] + padding[0] / img.width) * (img.width / img_new.width)
    coor[:, 1] = (coor[:, 1] + padding[1] / img.height) * (img.height/ img_new.height)
    return img_new, coor


def hflip(img, coor):
    coor = coor.copy()
    coor[:, 0] = 1 - coor[:, 0]
    return TF.hflip(img), coor


def vflip(img, coor):
    coor = coor.copy()
    coor[:, 1] = 1 - coor[:, 1]
    return TF.vflip(img), coor


def resize(img, coor, size, interpolation=Image.BILINEAR):
    # Assuming coordinates are 0/1-normalized.
    return TF.resize(img, size, interpolation=interpolation), coor


def resized_crop(img, coor, i, j, h, w, size, interpolation=Image.BILINEAR):
    img, coor = crop(img, coor, i, j, h, w)
    img, coor = resize(img, coor, size, interpolation)
    return img, coor


def refresh_valid(img, coor, force=False):
    if coor.shape[1] == 2:
        if force:
            coor = np.concatenate([coor, np.ones_like(coor[:, 0])], axis=1)
        else:
            return img, coor
    assert coor.shape[1] == 3, 'Support only (x, y, valid) or (x, y) typed coordinates.'
    out = []
    for x, y, v in coor:
        valid = (v == 1) and (x >= 0) and (x < img.width) and (y >= 0) and (y < img.height)
        if valid:
            out.append((x, y, v))
        else:
            out.append((0., 0., 0.))
    return img, np.array(out, dtype='float32')


def rotate(img, coor, angle, resample, crop_, expand, center=None, translate=None):
    assert translate is None
    img_new = TF.rotate(img, angle, resample=resample, expand=expand, center=center)
    matrix, extra_crop = get_rotation_matrix(img, angle, crop_, expand, center, translate)

    _, coor = denormalize_coor(img, coor)
    for i in range(coor.shape[0]):
        coor[i, :2] = apply_affine_transform(*coor[i, :2], matrix)
    _, coor = normalize_coor(img_new, coor)

    if extra_crop is not None:
        img_new, coor = crop(img_new, coor, *extra_crop)
    return img_new, coor


def pad_multiple_of(img, coor, multiple, mode='constant', fill=0):
    h, w = img.height, img.width
    hh = h - h % multiple + multiple * int(h % multiple != 0)
    ww = w - w % multiple + multiple * int(w % multiple != 0)
    if h != hh or w != ww:
        return pad(img, coor, (0, 0, ww - w, hh - h), mode=mode, fill=fill)
    return img, coor


def get_rotation_matrix(image, angle, crop, expand, center, translate):
    w, h = image.size
    if translate is None:
        translate = (0, 0)
    if center is None:
        center = (w / 2.0, h / 2.0)

    angle = math.radians(angle % 360)

    matrix = [
        round(math.cos(angle), 15), round(math.sin(angle), 15), 0.0,
        round(-math.sin(angle), 15), round(math.cos(angle), 15), 0.0
    ]

    matrix[2], matrix[5] = apply_affine_transform(-center[0], -center[1], matrix)
    matrix[2] += center[0] + translate[0]
    matrix[5] += center[1] + translate[1]

    # print('debug', angle, translate, center, matrix, apply_affine_transform(0.5, 0.5, matrix))

    if crop or expand:
        xx = []
        yy = []
        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            x, y = apply_affine_transform(x, y, matrix)
            xx.append(x)
            yy.append(y)

        xx.sort()
        yy.sort()

    extra_crop = None

    if crop:
        assert not expand, 'Cannot use both expand and crop.'
        nw = int(math.ceil(xx[2]) - math.floor(xx[1]))
        nh = int(math.ceil(yy[2]) - math.floor(yy[1]))

        # CAUSION! extra_crop is of format (dy, dx, h, w)
        extra_crop = ((h - nh) // 2, (w - nw) // 2, nh, nw)

    if expand:
        nw = int(math.ceil(xx[3]) - math.floor(xx[0]))
        nh = int(math.ceil(yy[3]) - math.floor(yy[0]))

        matrix[2] += (nw - w) / 2.
        matrix[5] += (nh - h) / 2.

    return matrix, extra_crop


def apply_affine_transform(x, y, matrix):
    (a, b, c, d, e, f) = matrix
    return a*x + b*y + c, d*x + e*y + f

