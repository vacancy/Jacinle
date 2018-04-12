# -*- coding: utf-8 -*-
# File   : variable.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 24/01/2018
# 
# This file is part of Jacinle.

import numpy as np
import torch
from torch.autograd import Variable

__all__ = ['var_with', 'new_var_with', 'var_from_list']


def var_with(obj, ref):
    if ref.is_cuda:
        obj = obj.cuda()
    if not isinstance(obj, Variable) and isinstance(ref, Variable):
        obj = Variable(obj)
    return obj


def new_var_with(obj, *args, **kwargs):
    is_variable = False
    if isinstance(obj, Variable):
        is_variable = True
        obj = obj.data
    res = obj.new(*args, **kwargs)
    if is_variable:
        res = Variable(res)
    return res


def var_from_list(value, dtype='float32', ref=None):
    value = np.array(value, dtype=dtype)
    value = torch.from_numpy(value)
    if ref is not None:
        return var_with(value, ref)
    return Variable(value)
