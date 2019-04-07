#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : _utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/16/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import math


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

        # CAUSION! extra_crop is of format (dx, dw, w, h)
        extra_crop = ((w - nw) // 2, (h - nh) // 2, nw, nh)

    if expand:
        nw = int(math.ceil(xx[3]) - math.floor(xx[0]))
        nh = int(math.ceil(yy[3]) - math.floor(yy[0]))

        matrix[2] += (nw - w) / 2.
        matrix[5] += (nh - h) / 2.

    return matrix, extra_crop


def apply_affine_transform(x, y, matrix):
    (a, b, c, d, e, f) = matrix
    return a*x + b*y + c, d*x + e*y + f


def get_size_multiple_of(h, w, multiple, residual):
    def _gen(x):
        actual = x % multiple
        if actual == residual:
            pass
        elif actual > residual:
            x += multiple + residual - actual
        elif actual < residual:
            x += residual - actual
        return x

    return _gen(h), _gen(w)

