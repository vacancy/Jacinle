# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/03/2018
# 
# This file is part of Jacinle.

from PIL import Image
import numpy as np

from torchvision.transforms import functional as TF
from jacinle.utils.argument import get_2dshape


def normalize_coor(img, coor):
    coor = coor.copy()
    coor[:, 0] /= img.width
    coor[:, 1] /= img.height
    return img, coor


def crop(img, coor, i, j, h, w):
    coor = coor.copy()

    coor[:, 0] = (coor[:, 0] - j / img.width) * (img.width / w)
    coor[:, 1] = (coor[:, 1] - i / img.height) * (img.height / h)
    return TF.crop(img, i, j, h, w), coor


def center_crop(img, coor, output_size):
    output_size = get_2dshape(output_size)
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, coor, i, j, th, tw)


def pad(img, coor, padding, fill=0):
    img_new = TF.pad(img, padding, fill=fill)
    coor = coor.coor.copy()
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    else:
        assert len(padding) == 4

    coor[:, 0] = (coor[:, 0] + padding[0] / img.width) * (img.width / img_new.width)
    coor[:, 1] = (coor[:, 1] + padding[1] / img.height) * (img.height/ img_new.height)
    return img_new, coor


def hflip(img, coor):
    coor = coor.copy()
    coor[:, 0] = 1 - coor[:, 0]
    return TF.hflip(img), coor


def vflip(img, coor):
    if random.random() < 0.5:
        return TF.vflip(img), coor
    return img, coor


def resize(img, coor, size, interpolation=Image.BILINEAR):
    # Assuming coordinates are 0/1-normalized.
    return TF.resize(img, size, interpolation=interpolation), coor


def resized_crop(img, coor, i, j, h, w, size, interpolation=Image.BILINEAR):
    img, coor = crop(img, coor, i, j, h, w)
    img, coor = resize(img, coor, size, interpolation)
    return img, coor


def refresh_valid(img, coor):
    out = []
    for x, y, v in coor:
        valid = (v == 1) and (x >= 0) and (x < 1) and (y >= 0) and (y < 1)
        if valid:
            out.append((x, y, v))
        else:
            out.append((0., 0., 0.))
    return img, np.array(out, dtype='float32')


def rotate(img, coor, angle, resample, expand, center):
    assert angle == 0
    return img, coor

