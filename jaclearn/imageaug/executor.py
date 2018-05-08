#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : executor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import functools
import copy

import jacinle.random as random


class AugmentorExecutorBase(object):
    def __init__(self, *, random_order=False):
        self._augmentors = []
        self._random_order = random_order

    def f(self, augmentor, **kwargs):
        partial = functools.partial(augmentor, **kwargs)
        self._augmentors.append(partial)
        return self

    def __call__(self, *args, **kwargs):
        if not self._random_order:
            augmentors = self._augmentors
        else:
            augmentors = copy.copy(self._augmentors)
            augmentors = random.shuffle(augmentors)
        return self._augment(augmentors, *args, **kwargs)

    def _augment(self, augmentors, *args, **kwargs):
        raise NotImplementedError()


class ImageAugmentorExecutor(AugmentorExecutorBase):
    def _augment(self, augmentors, img):
        for f in augmentors:
            img = f(img)
        return img


class ImageCoordAugmentorExecutor(AugmentorExecutorBase):
    def _augment(self, augmentors, img, coord=None):
        original_coord = coord
        for f in augmentors:
            res = f(img, coord=coord)
            if type(res) is tuple:
                img, coord = res
            else:
                img, coord = res, coord
        if original_coord is None:
            return img
        return img, coord
