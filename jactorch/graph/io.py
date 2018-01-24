# -*- coding: utf-8 -*-
# File   : io.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 24/01/2018
# 
# This file is part of Jacinle.

import collections

import numpy as np
import torch
from torch.autograd import Variable


def mark_volatile(obj):
    if torch.is_tensor(obj):
        obj = Variable(obj)
    if isinstance(obj, Variable):
        obj.volatile = True
        return obj
    elif isinstance(obj, collections.Mapping):
        return {k: mark_volatile(o) for k, o in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [mark_volatile(o) for o in obj]
    else:
        return obj


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    if isinstance(obj, (tuple, list)):
        return [as_variable(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: as_variable(v) for k, v in obj.items()}
    else:
        return Variable(obj)


def as_numpy(obj):
    if isinstance(obj, (tuple, list)):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)


def as_float(obj):
    if isinstance(obj, (tuple, list)):
        return [as_float(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: as_float(v) for k, v in obj.items()}
    else:
        arr = as_numpy(obj)
        assert arr.size == 1
        return float(arr)
