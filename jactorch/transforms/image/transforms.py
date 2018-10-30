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
        self.padding = padding
        self.mode = mode
        self.fill = fill

    def __call__(self, img):
        return F.pad(img, self.padding, mode=self.mode, fill=self.fill)


class PadMultipleOf(object):
    def __init__(self, multiple, mode='constant', fill=0):
        self.multiple = multiple
        self.mode = mode
        self.fill = fill

    def __call__(self, img):
        return F.pad_multiple_of(img, self.multiple, mode=self.mode, fill=self.fill)
