# -*- coding: utf-8 -*-
# File   : variable.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 24/01/2018
# 
# This file is part of Jacinle.

from torch.autograd import Variable


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
