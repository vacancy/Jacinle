#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : images.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from PIL import ImageOps

__all__ = ['PadMultipleOf']


class PadMultipleOf(object):
    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, img):
        h, w = img.height, img.width
        hh = h - h % self.multiple + self.multiple * int(h % self.multiple == 0)
        ww = w - w % self.multiple + self.multiple * int(w % self.multiple == 0)
        if h != hh or w != ww:
            return ImageOps.expand(img, (0, 0, ww - w, hh - h))
        return img

