#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : init.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/25/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os
import sys
import resource


def release_syslim():
    sys.setrecursionlimit(1000000)
    try:
        slim = 65536 * 1024
        resource.setrlimit(resource.RLIMIT_STACK, (slim, slim))
    except ValueError:
        pass


def tune_opencv():
    os.environ['OPENCV_OPENCL_RUNTIME'] = ''


def init_main():
    release_syslim()
    tune_opencv()
