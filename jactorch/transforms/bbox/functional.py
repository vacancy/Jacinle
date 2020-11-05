#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/03/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from PIL import Image
import numpy as np

from torchvision.transforms import functional as TF
from jacinle.utils.argument import get_2dshape


def normalize_bbox(img, bbox):
    """
    Normalize a bounding box.

    Args:
        img: (array): write your description
        bbox: (array): write your description
    """
    bbox = bbox.copy()
    bbox[:, 0] /= img.width
    bbox[:, 1] /= img.height
    bbox[:, 2] /= img.width
    bbox[:, 3] /= img.height
    return img, bbox


def denormalize_bbox(img, bbox):
    """
    Denormalize the image

    Args:
        img: (array): write your description
        bbox: (array): write your description
    """
    bbox = bbox.copy()
    bbox[:, 0] *= img.width
    bbox[:, 1] *= img.height
    bbox[:, 2] *= img.width
    bbox[:, 3] *= img.height
    return img, bbox


def crop(img, bbox, i, j, h, w):
    """
    Crops a bounding to fit.

    Args:
        img: (array): write your description
        bbox: (array): write your description
        i: (array): write your description
        j: (array): write your description
        h: (array): write your description
        w: (array): write your description
    """
    bbox = bbox.copy()

    bbox[:, 0] = (bbox[:, 0] - j / img.width) * (img.width / w)
    bbox[:, 1] = (bbox[:, 1] - i / img.height) * (img.height / h)
    bbox[:, 2] = (bbox[:, 2] - j / img.width) * (img.width / w)
    bbox[:, 3] = (bbox[:, 3] - i / img.height) * (img.height / h)
    return TF.crop(img, i, j, h, w), bbox


def center_crop(img, bbox, output_size):
    """
    Crops an image to the image

    Args:
        img: (array): write your description
        bbox: (array): write your description
        output_size: (int): write your description
    """
    output_size = get_2dshape(output_size)
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, bbox, i, j, th, tw)


def pad(img, bbox, padding, fill=0):
    """
    Pad a bounding box

    Args:
        img: (array): write your description
        bbox: (array): write your description
        padding: (float): write your description
        fill: (str): write your description
    """
    img_new = TF.pad(img, padding, fill=fill)
    bbox = bbox.copy()
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    else:
        assert len(padding) == 4

    bbox[:, 0] = (bbox[:, 0] + padding[0] / img.width) * (img.width / img_new.width)
    bbox[:, 1] = (bbox[:, 1] + padding[1] / img.height) * (img.height/ img_new.height)
    bbox[:, 2] = (bbox[:, 2] + padding[0] / img.width) * (img.width / img_new.width)
    bbox[:, 3] = (bbox[:, 3] + padding[1] / img.height) * (img.height/ img_new.height)
    return img_new, bbox


def hflip(img, bbox):
    """
    Hflipy.

    Args:
        img: (array): write your description
        bbox: (array): write your description
    """
    bbox = bbox.copy()
    bbox[:, 0] = 1 - bbox[:, 0]
    bbox[:, 2] = 1 - bbox[:, 2]
    return TF.hflip(img), bbox


def vflip(img, bbox):
    """
    Flip a bounding box

    Args:
        img: (array): write your description
        bbox: (array): write your description
    """
    bbox = bbox.copy()
    bbox[:, 1] = 1 - bbox[:, 1]
    bbox[:, 3] = 1 - bbox[:, 3]
    return TF.vflip(img), bbox


def resize(img, bbox, size, interpolation=Image.BILINEAR):
    """
    Resize an image.

    Args:
        img: (array): write your description
        bbox: (array): write your description
        size: (int): write your description
        interpolation: (int): write your description
        Image: (array): write your description
        BILINEAR: (array): write your description
    """
    # Assuming bboxdinates are 0/1-normalized.
    return TF.resize(img, size, interpolation=interpolation), bbox


def resized_crop(img, bbox, i, j, h, w, size, interpolation=Image.BILINEAR):
    """
    Resize an image.

    Args:
        img: (array): write your description
        bbox: (todo): write your description
        i: (array): write your description
        j: (array): write your description
        h: (array): write your description
        w: (array): write your description
        size: (int): write your description
        interpolation: (str): write your description
        Image: (array): write your description
        BILINEAR: (str): write your description
    """
    img, bbox = crop(img, bbox, i, j, h, w)
    img, bbox = resize(img, bbox, size, interpolation)
    return img, bbox


def rotate(img, bbox, angle, resample, expand, center):
    """
    Rotate the image

    Args:
        img: (array): write your description
        bbox: (int): write your description
        angle: (float): write your description
        resample: (int): write your description
        expand: (bool): write your description
        center: (float): write your description
    """
    assert angle == 0
    return img, bbox


def pad_multiple_of(img, coor, multiple, fill=0):
    """
    Pad image with pad

    Args:
        img: (array): write your description
        coor: (todo): write your description
        multiple: (bool): write your description
        fill: (str): write your description
    """
    h, w = img.height, img.width
    hh = h - h % multiple + multiple * int(h % multiple == 0)
    ww = w - w % multiple + multiple * int(w % multiple == 0)
    if h != hh or w != ww:
        return pad(img, coor, (0, 0, ww - w, hh - h), fill=fill)
    return img, coor

