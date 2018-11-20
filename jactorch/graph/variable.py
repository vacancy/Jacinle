#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : variable.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np

import torch
from torch.autograd import Variable

from jacinle.utils.deprecated import deprecated

__all__ = ['var_with', 'new_var_with', 'var_from_list']


@deprecated
def var_with(obj, ref):
    """DEPRECATED(Jiayuan Mao): var_with has been deprecated and will be removed by 10/23/2018; please use device=ref.device instead."""
    if ref.is_cuda:
        obj = obj.cuda()
    if not isinstance(obj, Variable) and isinstance(ref, Variable):
        obj = Variable(obj)
    return obj


@deprecated
def new_var_with(obj, *args, **kwargs):
    """DEPRECATED(Jiayuan Mao): new_var_with has been deprecated and will be removed by 10/23/2018; please use tensor.new instead."""
    is_variable = False
    if isinstance(obj, Variable):
        is_variable = True
        obj = obj.data
    res = obj.new(*args, **kwargs)
    if is_variable:
        res = Variable(res)
    return res


@deprecated
def var_from_list(value, dtype='float32', ref=None):
    """DEPRECATED(Jiayuan Mao): var_from_list has been deprecated and will be removed by 10/23/2018; please use torch.tensor instead."""
    value = np.array(value, dtype=dtype)
    value = torch.from_numpy(value)
    if ref is not None:
        return var_with(value, ref)
    return Variable(value)
