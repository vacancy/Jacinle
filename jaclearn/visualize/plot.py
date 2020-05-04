#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plot.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import io as _io
import numpy as np
from PIL import Image

from jacinle.image.backend import cv2, Image, opencv_only, pil_only
from jacinle.utils.enum import JacEnum

__all__ = ['plot2opencv', 'plot2pil', 'heatmap2pil']


@opencv_only
def plot2opencv(fig):
    """Convert a pyplot instance to image"""

    buf = _io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    rawbuf = np.frombuffer(buf.getvalue(), dtype='uint8')
    im = cv2.imdecode(rawbuf, cv2.IMREAD_COLOR)
    buf.close()
    return im


@pil_only
def plot2pil(fig):
    canvas = fig.canvas
    canvas.draw()
    pil = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    return pil


class HeatmapNormalization(JacEnum):
    NONE = 'none'
    RANGE = 'range'
    INSTANCE = 'instance'


@pil_only
def heatmap2pil(heatmap, normalization='none', minval=0, maxval=1):
    heatmap = _to_numpy(heatmap)

    normalization = HeatmapNormalization.from_string(normalization)
    if normalization is HeatmapNormalization.NONE:
        pass
    elif normalization is HeatmapNormalization.MINMAX:
        heatmap = (heatmap - minval) / (maxval - minval)
    elif normalization is HeatmapNormalization.INSTANCE:
        minval, maxval = heatmap.min(), heatmap.max()
        heatmap = (heatmap - minval) / (maxval - minval)
    else:
        raise ValueError('Unknown heatmap normalization: {}.'.format(normalization))

    return Image.fromarray((heatmap * 255).astype('uint8'))


def _to_numpy(obj):
    # NB(Jiayuan Mao @ 05/03): hack for pytorch tensors.
    if hasattr(obj, 'cpu'):
        obj = obj.cpu()
    return np.array(obj)

