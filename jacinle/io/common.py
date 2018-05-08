#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : common.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os.path as osp
import math
import contextlib

__all__ = ['get_ext', 'fsize_format', 'auto_close']


unit_list = list(zip(['bytes', 'kB', 'MB', 'GB', 'TB', 'PB'], [0, 0, 1, 2, 2, 2]))


def get_ext(fname, match_first=False):
    if match_first:
        fname = osp.split(fname)[1]
        return fname[fname.find('.'):]
    else:
        return osp.splitext(fname)[1]


def fsize_format(num):
    """Human readable file size."""
    # from http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size

    if num == 0:
        return '0 bytes'
    if num == 1:
        return '1 byte'

    exponent = min(int(math.log(num, 1024)), len(unit_list) - 1)
    quotient = float(num) / 1024**exponent
    unit, num_decimals = unit_list[exponent]
    format_string = '{:.%sf} {}' % num_decimals
    return format_string.format(quotient, unit)


@contextlib.contextmanager
def auto_close(file):
    yield file
    file.close()
