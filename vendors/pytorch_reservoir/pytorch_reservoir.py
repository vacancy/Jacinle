#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/11/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import sys
import os.path as osp
from jacinle.jit.cext import auto_travis
from jacinle.utils.cache import cached_result


@cached_result
def _load_extension():
    auto_travis(__file__)
    sys.path.insert(0, osp.join(osp.dirname(__file__), 'torch_sampling'))
    import torch_sampling
    sys.path = sys.path[1:]
    return torch_sampling


# TODO(Jiayuan Mao @ 04/11): add the signature.
def choice(*args, **kwargs):
    return _load_extension().choice(*args, **kwargs)


def reservoir_sampling(*args, **kwargs):
    return _load_extension().reservoir_sampling(*args, **kwargs)

