#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : photography.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np

import jacinle.random as random
from jacinle.image import imgproc

__all__ = [
    'grayscale_augment',
    'brightness_augment', 'contrast_augment', 'saturation_augment',
    'color_augment_pack',
    'lighting_augment'
]


def grayscale_augment(img, prob=0.5):
    """
    Grays an image with a constant.

    Args:
        img: (array): write your description
        prob: (todo): write your description
    """
    if random.rand() <= prob:
        return imgproc.grayscale(img)
    return img


def brightness_augment(img, val):
    """
    Brightness brightness.

    Args:
        img: (array): write your description
        val: (float): write your description
    """
    alpha = 1. + val * (random.rand() * 2 - 1)
    return imgproc.brightness(img, alpha)


def contrast_augment(img, val):
    """
    Contrast an image with an image.

    Args:
        img: (array): write your description
        val: (str): write your description
    """
    alpha = 1. + val * (random.rand() * 2 - 1)
    return imgproc.contrast(img, alpha)


def saturation_augment(img, val):
    """
    Generate an image with a constant.

    Args:
        img: (array): write your description
        val: (float): write your description
    """
    alpha = 1. + val * (random.rand() * 2 - 1)
    return imgproc.saturation(img, alpha)


def color_augment_pack(img, brightness, contrast, saturation):
    """
    Shuffle an rgb image with colors.

    Args:
        img: (array): write your description
        brightness: (int): write your description
        contrast: (todo): write your description
        saturation: (int): write your description
    """
    augmentors = list(zip(
        (brightness_augment, contrast_augment, saturation_augment),
        (brightness, contrast, saturation)
    ))
    random.shuffle(augmentors)

    for f, val in augmentors:
        img = f(img, val)
    return img


def lighting_augment(img, std, eigval=None, eigvec=None):
    """
    Return an eigenvalue from eigenvalue

    Args:
        img: (array): write your description
        std: (todo): write your description
        eigval: (todo): write your description
        eigvec: (int): write your description
    """
    eigval = eigval or np.array([0.2175, 0.0188, 0.0045])
    eigvec = eigvec or np.array([
        [-0.5836, -0.6948, 0.4203],
        [-0.5808, -0.0045, -0.8140],
        [-0.5675, 0.7192, 0.4009]
    ])
    if std == 0:
        return img

    alpha = random.randn(3) * std
    bgr = eigvec * alpha.reshape(1, 3) * eigval.reshape(1, 3)
    bgr = bgr.sum(axis=1).reshape(1, 1, 3)
    img = img + bgr

    return img