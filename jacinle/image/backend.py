#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : backend.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time
import functools
import numpy as np
from jacinle.logging import get_logger

logger = get_logger(__file__)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

if cv2 is None:
    if Image is not None:
        logger.warning('Fail to import OpenCV; use PIL library.')
    else:
        logger.error('Can not find either PIL OpenCV; you can not use most function in tartist.image.')


FORCE_PIL_BGR = True


def opencv_or_pil(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        if cv2 is None and Image is None:
            assert False, 'Call {} without OpenCV or PIL.'.format(func)
        return func(*args, **kwargs)
    return new_func


def opencv_only(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        if cv2 is None:
            assert False, 'Call {} without OpenCV.'.format(func)
        return func(*args, **kwargs)
    return new_func


def pil_only(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        if Image is None:
            assert False, 'Call {} without PIL.'.format(func)
        return func(*args, **kwargs)
    return new_func


def pil_img2nd(image, require_chl3=True):
    nd = np.array(image)
    if FORCE_PIL_BGR:
        nd = nd[:, :, ::-1]
    if require_chl3 and len(nd.shape) == 2:
        return nd[:, :, np.newaxis]
    return nd


def pil_nd2img(image):
    if FORCE_PIL_BGR:
        image = image[:, :, ::-1]
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return Image.fromarray(image)


@opencv_or_pil
def imread(path):
    if cv2:
        return cv2.imread(path)
    else:
        image = Image.open(path)
        return pil_img2nd(image)


@opencv_or_pil
def imwrite(path, image):
    if cv2:
        return cv2.imwrite(path, image)
    else:
        image = pil_nd2img(image)
        return image.save(path)


@opencv_or_pil
def imshow(title, image):
    if cv2:
        cv2.imshow(title, image)
        cv2.waitKey(0)
    else:
        image = pil_nd2img(image)
        image.show(title)
        time.sleep(0.5)


@opencv_or_pil
def resize(image, dsize, interpolation='LINEAR'):
    assert interpolation in ('NEAREST', 'LINEAR', 'CUBIC', 'LANCZOS4')

    dsize = tuple(map(int, dsize))
    assert len(dsize) == 2
    if cv2:
        interpolation = getattr(cv2, 'INTER_' + interpolation)
        return cv2.resize(image, dsize, interpolation=interpolation)
    else:
        if interpolation == 'NEAREST':
            interpolation = Image.NEAREST
        elif interpolation == 'LANCZOS4':
            interpolation = Image.LANCZOS
        else:
            interpolation = getattr(Image, 'BI' + interpolation)

        image = pil_nd2img(image)
        image = image.resize(dsize, resample=interpolation)
        return pil_img2nd(image)
