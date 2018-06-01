#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : imgproc.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import math
import functools

import numpy as np

from jacinle.utils.argument import get_2dshape
from jacinle.utils.enum import JacEnum

from . import backend

__all__ = [
    'resize', 'resize_wh', 'resize_scale', 'resize_scale_wh', 'resize_minmax',
    'crop', 'center_crop', 'leftup_crop',
    'dimshuffle',
    'clip', 'clip_decorator',
    'grayscale',
    'brightness', 'contrast', 'saturation'
]


def _get_crop2d_rest(img, target_shape):
    source_shape = img.shape[:2]
    target_shape = get_2dshape(target_shape)
    rest_shape = source_shape[0] - target_shape[0], source_shape[1] - target_shape[1]
    assert rest_shape[0] >= 0 and rest_shape[1] >= 0
    return rest_shape


def _crop2d(img, start, size):
    return img[start[0]:start[0] + size[0], start[1]:start[1] + size[1]]


def resize(img, size, interpolation='LINEAR'):
    size = get_2dshape(size)
    return backend.resize(img, (size[1], size[0]), interpolation=interpolation)


def resize_wh(img, size_wh, interpolation='LINEAR'):
    size_wh = get_2dshape(size_wh)
    return backend.resize(img, size_wh, interpolation=interpolation)


def resize_scale(img, scale, interpolation='LINEAR'):
    scale = get_2dshape(scale, type=float)
    new_size = math.ceil(img.shape[0] * scale[0]), math.ceil(img.shape[1] * scale[1])
    return resize(img, new_size, interpolation=interpolation)


def resize_scale_wh(img, scale_wh, interpolation='LINEAR'):
    scale_wh = get_2dshape(scale_wh, type=float)
    return resize_scale(img, (scale_wh[1], scale_wh[0]), interpolation=interpolation)


def resize_minmax(img, min_dim, max_dim=None, interpolation='LINEAR'):
    if max_dim is None:
        max_dim = min_dim
    min_dim, max_dim = min(min_dim, max_dim), max(min_dim, max_dim)

    h, w = img.shape[:2]
    short, long = min(h, w), max(h, w)
    scale = min_dim / short
    scale = min(max_dim / long, scale)
    return resize_scale(img, scale, interpolation=interpolation)


def crop(image, l, t, w, h, extra_crop=None):
    if extra_crop is not None and extra_crop != 1:
        new_w, new_h = round(w * extra_crop), round(h * extra_crop)
        l -= (new_w - w) // 2
        t -= (new_h - h) // 2
        w, h = new_w, new_h

    im_h, im_w = image.shape[0:2]
    w, h = int(round(w)), int(round(h))
    l, t = int(math.floor(l)), int(math.floor(t))
    # range is expected to be image[t:t+h, l:l+w] now.

    ex_l, ex_t, ex_w, ex_h = l, t, w, h
    delta_l, delta_t = 0, 0
    if ex_l < 0:
        ex_l = 0
        delta_l = ex_l - l
        ex_w -= delta_l
    if ex_t < 0:
        ex_t = 0
        delta_t = ex_t - t
        ex_h -= delta_t
    if ex_l + ex_w > im_w:
        ex_w = im_w - ex_l
    if ex_t + ex_h > im_h:
        ex_h = im_h - ex_t

    result = np.zeros(shape=(h, w) + image.shape[2:], dtype=image.dtype)
    result[delta_t:delta_t + ex_h, delta_l:delta_l + ex_w] = image[ex_t:ex_t + ex_h, ex_l:ex_l + ex_w]
    return result


def center_crop(img, target_shape):
    """ center crop """
    target_shape = get_2dshape(target_shape)
    rest = _get_crop2d_rest(img, target_shape)
    start = rest[0] // 2, rest[1] // 2

    return _crop2d(img, start, target_shape)


def leftup_crop(img, target_shape):
    """ left-up crop """
    start = 0, 0
    target_shape = get_2dshape(target_shape)

    return _crop2d(img, start, target_shape)


class ShuffleType(JacEnum):
    CHANNEL_FIRST = 'channel_first'
    CHANNEL_LAST = 'channel_last'


def dimshuffle(img, shuffle_type):
    shuffle_type = ShuffleType.from_string(shuffle_type)
    assert len(img.shape) in (2, 3, 4), 'Image should be of dims 2, 3 or 4'

    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        if shuffle_type == ShuffleType.CHANNEL_FIRST:
            return np.transpose(img, (2, 0, 1))
        else:
            return np.transpose(img, (1, 2, 0))
    else:  # len(img.shape) == 4:
        if shuffle_type == ShuffleType.CHANNEL_FIRST:
            return np.transpose(img, (0, 3, 1, 2))
        else:
            return np.transpose(img, (0, 2, 3, 1))


def clip(img):
    return np.minimum(255, np.maximum(0, img))


def clip_decorator(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        img = func(*args, **kwargs)
        return clip(img)

    return new_func


def grayscale(img):
    assert len(img.shape) == 3 and img.shape[2] == 3
    w = np.array([0.114, 0.587, 0.299]).reshape(1, 1, 3)
    img = (img * w).sum(axis=2, keepdims=True)
    return img


@clip_decorator
def brightness(img, alpha):
    return img * alpha


@clip_decorator
def contrast(img, alpha):
    gs = grayscale(img)
    gs[:] = gs.mean()
    img = img * alpha + gs * (1 - alpha)
    return img


@clip_decorator
def saturation(img, alpha):
    gs = grayscale(img)
    img = img * alpha + gs * (1 - alpha)
    return img
