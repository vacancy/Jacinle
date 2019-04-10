#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : peak.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/07/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
Extracting peaks from feature maps: inspired by `skimage.features.peak`.
"""

import torch
from .smooth import maximum_smooth

__all__ = ['peak_local_max']


def _get_high_intensity_peaks(is_peak, nr_peaks, feature_maps):
    peaks = torch.nonzero(is_peak)
    if nr_peaks is not None:
        # TODO(Jiayuan Mao @ 04/07): select the top-K peaks.
        raise NotImplementedError()
    return is_peak, peaks


def peak_local_max(feature_maps, radius, exclude_border=True):
    if type(exclude_border) is bool:
        if exclude_border:
            exclude_border = radius
        else:
            exclude_border = 0

    max_fmaps = maximum_smooth(feature_maps, 2 * radius + 1)
    is_peak = max_fmaps == feature_maps

    if exclude_border > 0:
        is_peak[..., :exclude_border, :] = 0
        is_peak[..., -exclude_border:, :] = 0
        is_peak[..., :, :exclude_border] = 0
        is_peak[..., :, -exclude_border:] = 0

    return _get_high_intensity_peaks(is_peak, None, None)

