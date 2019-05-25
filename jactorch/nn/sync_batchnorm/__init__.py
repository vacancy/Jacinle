#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.utils.vendor import has_vendor, requires_vendors

__all__ = [
    'SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d',
    'patch_sync_batchnorm', 'convert_sync_batchnorm'
]


if has_vendor('sync_batchnorm'):
    from sync_batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
    from sync_batchnorm import patch_sync_batchnorm, convert_model as convert_sync_batchnorm
else:
    from jacinle.utils.meta import make_dummy_func
    SynchronizedBatchNorm1d = requires_vendors('sync_batchnorm')(make_dummy_func())
    SynchronizedBatchNorm2d = requires_vendors('sync_batchnorm')(make_dummy_func())
    SynchronizedBatchNorm3d = requires_vendors('sync_batchnorm')(make_dummy_func())
    patch_sync_batchnorm = requires_vendors('sync_batchnorm')(make_dummy_func())
    convert_sync_batchnorm = requires_vendors('sync_batchnorm')(make_dummy_func())

