#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : transforms.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from . import functional as F

__all__ = ['Pad', 'PadMultipleOf']


class Pad(object):
    def __init__(self, padding, mode='constant', fill=0):
        """
        Set padding.

        Args:
            self: (todo): write your description
            padding: (str): write your description
            mode: (todo): write your description
            fill: (str): write your description
        """
        self.padding = padding
        self.mode = mode
        self.fill = fill

    def __call__(self, img):
        """
        Call this image.

        Args:
            self: (todo): write your description
            img: (todo): write your description
        """
        return F.pad(img, self.padding, mode=self.mode, fill=self.fill)


class PadMultipleOf(object):
    def __init__(self, multiple, mode='constant', fill=0):
        """
        Initialize the mode.

        Args:
            self: (todo): write your description
            multiple: (todo): write your description
            mode: (todo): write your description
            fill: (str): write your description
        """
        self.multiple = multiple
        self.mode = mode
        self.fill = fill

    def __call__(self, img):
        """
        Pad the image.

        Args:
            self: (todo): write your description
            img: (todo): write your description
        """
        return F.pad_multiple_of(img, self.multiple, mode=self.mode, fill=self.fill)
