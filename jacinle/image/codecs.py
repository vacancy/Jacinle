#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : codecs.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from .backend import cv2, opencv_only
import numpy as np

__all__ = ['jpeg_encode', 'png_encode', 'imdecode']


@opencv_only
def jpeg_encode(img, quality=90):
    """Encode the image with JPEG encoder.

    Args:
        img (:class:`numpy.ndarray`): uint8 color image array
        quality (int): quality for JPEG compression

    Returns bytes: encoded image data
    """
    return cv2.imencode('.jpg', img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tostring()


@opencv_only
def png_encode(input, compress_level=3):
    """
    Encode the image with PNG encoder.

    Args:
        img (:class:`numpy.ndarray`): uint8 color image array
        quality (int): quality for JPEG compression

    Returns bytes: encoded image data
    """
    assert len(input.shape) == 3 and input.shape[2] in [3, 4]
    assert input.dtype == np.uint8
    assert isinstance(compress_level, int) and 0 <= compress_level <= 9
    enc = cv2.imencode('.png', input,
                       [int(cv2.IMWRITE_PNG_COMPRESSION), compress_level])
    return enc[1].tostring()


@opencv_only
def imdecode(data, *, require_chl3=True, require_alpha=False):
    """Decode images in common formats (jpg, png, etc.).

    Args:
        data (bytes): encoded image data
        require_chl3: whether to convert gray image to 3-channel BGR image
        require_alpha: whether to add alpha channel to BGR image

    Returns: uint8 color image array
    """

    img = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img is not None, 'failed to decode'
    if img.ndim == 2 and require_chl3:
        img = img.reshape(img.shape + (1,))
    if img.shape[2] == 1 and require_chl3:
        img = np.tile(img, (1, 1, 3))
    if img.ndim == 3 and img.shape[2] == 3 and require_alpha:
        assert img.dtype == np.uint8
        img = np.concatenate([img, np.ones_like(img[:, :, :1]) * 255], axis=2)
    return img
