#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : imgio.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.


import os.path as osp

from . import backend
from .imgproc import dimshuffle


__all__ = ['imread', 'imwrite', 'imshow']


def imread(path, *, shuffle=False):
    if not osp.exists(path):
        return None
    i = backend.imread(path)
    if i is None:
        return None
    if shuffle:
        return dimshuffle(i, 'channel_first')
    return i


def imwrite(path, img, *, shuffle=False):
    if shuffle:
        img = dimshuffle(img, 'channel_last')
    backend.imwrite(path, img)


def imshow(title, img, *, shuffle=False):
    if shuffle:
        img = dimshuffle(img, 'channel_last')
    backend.imshow(title, img)
