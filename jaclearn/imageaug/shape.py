#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : shape.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import itertools
import collections

import jacinle.random as random
from jacinle.image import imgproc
from jacinle.utils.argument import get_2dshape

__all__ = [
    'random_crop',
    'random_crop_random_shape',
    'random_crop_and_resize',
    'random_size_crop',
    'horizontal_flip_augment'
]


def _rand_2dshape(upper_bound, lower_bound=None):
    lower_bound = lower_bound or (0, ) * len(upper_bound)
    return tuple(itertools.starmap(random.randint, zip(lower_bound, upper_bound)))


def random_crop(img, target_shape):
    """random crop a image. output size is target_shape"""
    rest = imgproc._get_crop2d_rest(img, target_shape)
    start = _rand_2dshape(rest)
    return imgproc._crop2d(img, start, target_shape)


def random_crop_random_shape(img, max_shape, min_shape=0):
    max_shape = get_2dshape(max_shape)
    min_shape = get_2dshape(min_shape)
    assert min_shape[0] < img.shape[0] < max_shape[0] and min_shape[1] < img.shape[1] < max_shape[1]

    tar_shape = _rand_2dshape(max_shape, lower_bound=min_shape)
    return random_crop(img, tar_shape)


def random_crop_and_resize(img, max_shape, target_shape, min_shape=0):
    target_shape = get_2dshape(target_shape)
    cropped = random_crop_random_shape(img, max_shape, min_shape=min_shape)
    return imgproc.resize(cropped, target_shape)


def random_size_crop(img, target_shape, area_range, aspect_ratio=None, contiguous_ar=False, *, nr_trial=10):
    """random size crop used for Facebook ImageNet data augmentation
    see https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua
    """

    target_shape = get_2dshape(target_shape)
    h, w = img.shape[:2]
    area = h * w
    area_range = area_range if isinstance(area, collections.Iterable) else (area_range, 1)

    if aspect_ratio is None:
        assert contiguous_ar == False
        aspect_ratio = [h / w]

    for i in range(nr_trial):
        target_area = random.uniform(area_range[0], area_range[1]) * area
        target_ar = random.choice(aspect_ratio)
        nw = int(round((target_area * target_ar) ** 0.5))
        nh = int(round((target_area / target_ar) ** 0.5))

        if random.rand() < 0.5:
                nh, nw = nw, nh

        if nh <= h and nw <= w:
            sx, sy = random.randint(w - nw + 1), random.randint(h - nh + 1)
            img = img[sy:sy + nh, sx:sx + nw]

            return imgproc.resize(img, target_shape)

    scale = min(*target_shape) / min(h, w)
    return imgproc.center_crop(imgproc.resize_scale(img, scale), target_shape)


def horizontal_flip_augment(img, prob):
    if random.rand() < prob:
        return img[:, ::-1]
    return img