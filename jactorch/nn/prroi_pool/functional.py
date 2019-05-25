#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/25/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.utils.vendor import has_vendor, requires_vendors

__all__ = ['prroi_pool2d']


if has_vendor('prroi_pool'):
    from prroi_pool.functional import prroi_pool2d
else:
    from jacinle.utils.meta import make_dummy_func
    prroi_pool2d = requires_vendors('prroi_pool')(make_dummy_func())

