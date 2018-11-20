#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/29/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from PIL import Image
import numpy as np

from torchvision.transforms import functional as TF


def pad(img, padding, mode='constant', fill=0):
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    else:
        assert len(padding) == 4

    if mode == 'constant':
        img_new = TF.pad(img, padding, fill=fill)
    else:
        np_padding = ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0))
        img_new = Image.fromarray(np.pad(
            np.array(img), np_padding, mode=mode
        ))

    return img_new


def pad_multiple_of(img, multiple, mode='constant', fill=0):
    h, w = img.height, img.width
    hh = h - h % multiple + multiple * int(h % multiple != 0)
    ww = w - w % multiple + multiple * int(w % multiple != 0)
    if h != hh or w != ww:
        return pad(img, (0, 0, ww - w, hh - h), mode=mode, fill=fill)
    return img

