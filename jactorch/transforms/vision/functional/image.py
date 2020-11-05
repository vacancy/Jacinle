#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : image.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/14/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.


from PIL import Image
import numpy as np

from torchvision.transforms import functional as TF

__all__ = [
    'to_tensor', 'to_pil_image',
    'normalize',
    'pad', 'crop', 'resize', 'hflip', 'vflip',
    'five_crop', 'ten_crop',
    'adjust_brightness', 'adjust_contrast', 'adjust_saturation', 'adjust_hue', 'adjust_gamma', 'to_grayscale',
    'rotate', 'affine'
]


to_tensor = TF.to_tensor
to_pil_image = TF.to_pil_image

normalize = TF.normalize


def pad(img, padding, mode='constant', fill=0):
    """
    Pad an image with padding padding.

    Args:
        img: (array): write your description
        padding: (float): write your description
        mode: (todo): write your description
        fill: (str): write your description
    """
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


def crop(img, x, y, w, h):
    """
    Crop the image

    Args:
        img: (array): write your description
        x: (array): write your description
        y: (array): write your description
        w: (array): write your description
        h: (array): write your description
    """
    return TF.crop(img, y, x, h, w)


center_crop = TF.center_crop
resize = TF.resize
hflip = TF.hflip
vflip = TF.vflip

five_crop = TF.five_crop
ten_crop = TF.ten_crop

to_grayscale = TF.to_grayscale

rotate = TF.rotate
affine = TF.affine

adjust_brightness = TF.adjust_brightness
adjust_contrast = TF.adjust_contrast
adjust_saturation = TF.adjust_saturation
adjust_hue = TF.adjust_hue
adjust_gamma = TF.adjust_gamma
