#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/25/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.utils.vendor import has_vendor, requires_vendors

__all__ = ['PrRoIPool2D']


if has_vendor('prroi_pool'):
    from prroi_pool import PrRoIPool2D
else:
    from jacinle.utils.meta import make_dummy_func
    PrRoIPool2D = requires_vendors('prroi_pool')(make_dummy_func())

