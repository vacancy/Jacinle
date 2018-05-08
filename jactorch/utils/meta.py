# -*- coding: utf-8 -*-
# File   : meta.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 24/01/2018
# 
# This file is part of Jacinle.

import six
import numpy as np

import torch

from jacinle.utils.deprecated import deprecated
from jacinle.utils.meta import stmap

SKIP_TYPES = six.string_types


def _mark_volatile(o):
    from torch.autograd import Variable
    if torch.is_tensor(o):
        o = Variable(o)
    if isinstance(o, Variable):
        o.volatile = True
    return o


@deprecated
def mark_volatile(obj):
    """DEPRECATED(Jiayuan Mao): mark_volatile has been deprecated and will be removed by 10/23/2018; please use torch.no_grad instead."""
    return stmap(_mark_volatile, obj)


def _as_tensor(o):
    from torch.autograd import Variable
    if isinstance(o, SKIP_TYPES):
        return o
    """DEPRECATED(Jiayuan Mao): variable cast has been deprecated and will be removed by 10/23/2018; please use o directly instead."""
    if isinstance(o, Variable):
        return o.data
    if torch.is_tensor(o):
        return o
    return torch.from_numpy(np.array(o))


def as_tensor(obj):
    return stmap(_as_tensor, obj)


def _as_variable(o):
    from torch.autograd import Variable
    if isinstance(o, SKIP_TYPES):
        return o
    if isinstance(o, Variable):
        return o
    if not torch.is_tensor(o):
        o = torch.from_numpy(np.array(o))
    return Variable(o)


@deprecated
def as_variable(obj):
    """DEPRECATED(Jiayuan Mao): as_variable has been deprecated and will be removed by 10/23/2018; please use as_tensor instead."""
    return stmap(_as_variable, obj)


def _as_numpy(o):
    from torch.autograd import Variable
    if isinstance(o, SKIP_TYPES):
        return o
    """DEPRECATED(Jiayuan Mao): variable cast has been deprecated and will be removed by 10/23/2018; please use o directly instead."""
    if isinstance(o, Variable):
        o = o.data
    if torch.is_tensor(o):
        return o.cpu().numpy()
    return np.array(o)


def as_numpy(obj):
    return stmap(_as_numpy, obj)


def _as_float(o):
    if isinstance(o, SKIP_TYPES):
        return o
    arr = as_numpy(o)
    assert arr.size == 1
    return float(arr)


def as_float(obj):
    return stmap(_as_float, obj)


def _as_cpu(o):
    from torch.autograd import Variable
    if isinstance(o, Variable) or torch.is_tensor(o):
        return o.cpu()
    return o


def as_cpu(obj):
    return stmap(_as_cpu, obj)


def _as_cuda(o):
    from torch.autograd import Variable
    if isinstance(o, Variable) or torch.is_tensor(o):
        return o.cuda()
    return o


def as_cuda(obj):
    return stmap(_as_cuda, obj)
