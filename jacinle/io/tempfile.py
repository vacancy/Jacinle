#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : tempfile.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import tempfile as tempfile_lib
import contextlib
import os

__all__ = ['tempfile']


@contextlib.contextmanager
def tempfile(mode='w+b', suffix='', prefix='tmp'):
    f = tempfile_lib.NamedTemporaryFile(mode, suffix=suffix, prefix=prefix, delete=False)
    yield f
    os.unlink(f.name)

