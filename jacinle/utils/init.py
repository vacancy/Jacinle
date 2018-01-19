# -*- coding: utf-8 -*-
# File   : init.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/25/17
#
# This file is part of Jacinle.

import os
import sys
import resource

__all__ = ['release_syslim', 'tune_opencv', 'tune_tensorflow', 'initialize_main']


def release_syslim():
    sys.setrecursionlimit(1000000)
    try:
        slim = 65536 * 1024
        resource.setrlimit(resource.RLIMIT_STACK, (slim, slim))
    except ValueError:
        pass


def tune_tensorflow():
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'  # issue#9339
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '3'   # use more warm-up


def tune_opencv():
    os.environ['OPENCV_OPENCL_RUNTIME'] = ''


def initialize_main():
    release_syslim()
    tune_tensorflow()
