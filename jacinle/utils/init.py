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
from jacinle.utils.env import jac_getenv, jac_is_debug, jac_getenv


def release_syslim():
    if jac_getenv('SYSLIM', default='n', type='bool'):
        sys.setrecursionlimit(1000000)
        try:
            slim = 65536 * 1024
            resource.setrlimit(resource.RLIMIT_STACK, (slim, slim))
        except ValueError:
            pass


def tune_opencv():
    os.environ['OPENCV_OPENCL_RUNTIME'] = ''


def enable_ipdb():
    if jac_is_debug():
        if jac_getenv('IMPORT_ALL', 'true', 'bool'):
            from jacinle.utils.debug import hook_exception_ipdb
            hook_exception_ipdb()


def init_main():
    release_syslim()
    tune_opencv()
    enable_ipdb()

