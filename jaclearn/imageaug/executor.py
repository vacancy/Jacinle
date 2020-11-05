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
        """
        Initialize the scaling.

        Args:
            self: (todo): write your description
            random_order: (int): write your description
        """
        self._augmentors = []
        self._random_order = random_order

    def f(self, augmentor, **kwargs):
        """
        Decorates function to callable.

        Args:
            self: (todo): write your description
            augmentor: (todo): write your description
        """
        partial = functools.partial(augmentor, **kwargs)
        self._augmentors.append(partial)
        return self

    def __call__(self, *args, **kwargs):
        """
        Return a new method.

        Args:
            self: (todo): write your description
        """
        if not self._random_order:
            augmentors = self._augmentors
        else:
            augmentors = copy.copy(self._augmentors)
            augmentors = random.shuffle(augmentors)
        return self._augment(augmentors, *args, **kwargs)

    def _augment(self, augmentors, *args, **kwargs):
        """
        Shortcut todo.

        Args:
            self: (todo): write your description
            augmentors: (todo): write your description
        """
        raise NotImplementedError()


class ImageAugmentorExecutor(AugmentorExecutorBase):
    def _augment(self, augmentors, img):
        """
        Return a list of the given image.

        Args:
            self: (todo): write your description
            augmentors: (todo): write your description
            img: (array): write your description
        """
        for f in augmentors:
            img = f(img)
        return img


class ImageCoordAugmentorExecutor(AugmentorExecutorBase):
    def _augment(self, augmentors, img, coord=None):
        """
        Return an image at the given coordinates.

        Args:
            self: (todo): write your description
            augmentors: (todo): write your description
            img: (array): write your description
            coord: (todo): write your description
        """
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
