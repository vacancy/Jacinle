#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : meta.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import six
import functools
import numpy as np

import torch

from jacinle.utils.deprecated import deprecated
from jacinle.utils.meta import stmap

SKIP_TYPES = six.string_types

__all__ = ['mark_volatile', 'as_tensor', 'as_variable', 'as_numpy', 'as_float', 'as_cuda', 'as_cpu', 'as_detached']


def _mark_volatile(o):
    """
    Mark tensor as a tensor.

    Args:
        o: (int): write your description
    """
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
    """
    Convert o to a tensor.

    Args:
        o: (todo): write your description
    """
    from torch.autograd import Variable
    if isinstance(o, SKIP_TYPES):
        return o
    if isinstance(o, Variable):
        return o
    if torch.is_tensor(o):
        return o
    return torch.from_numpy(np.array(o))


def as_tensor(obj):
    """
    Convert obj into a tensor.

    Args:
        obj: (todo): write your description
    """
    return stmap(_as_tensor, obj)


def _as_variable(o):
    """
    Convert an object as variable.

    Args:
        o: (array): write your description
    """
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
    """
    Convert an object to a numpy.

    Args:
        o: (array): write your description
    """
    from torch.autograd import Variable
    if isinstance(o, SKIP_TYPES):
        return o
    if isinstance(o, Variable):
        o = o
    if torch.is_tensor(o):
        return o.cpu().numpy()
    return np.array(o)


def as_numpy(obj):
    """
    Convert obj as_numpy. numpy.

    Args:
        obj: (todo): write your description
    """
    return stmap(_as_numpy, obj)


def _as_float(o):
    """
    Convert an object as a float.

    Args:
        o: (dict): write your description
    """
    if isinstance(o, SKIP_TYPES):
        return o
    if torch.is_tensor(o):
        return o.item()
    arr = as_numpy(o)
    assert arr.size == 1
    return float(arr)


def as_float(obj):
    """
    Convert obj to float. float.

    Args:
        obj: (todo): write your description
    """
    return stmap(_as_float, obj)


def _as_cpu(o):
    """
    Return the cpu astorch variable.

    Args:
        o: (todo): write your description
    """
    from torch.autograd import Variable
    if isinstance(o, Variable) or torch.is_tensor(o):
        return o.cpu()
    return o


def as_cpu(obj):
    """
    Convert obj as_cpu.

    Args:
        obj: (todo): write your description
    """
    return stmap(_as_cpu, obj)


def _as_cuda(o):
    """
    Convert an object as cuda object.

    Args:
        o: (todo): write your description
    """
    from torch.autograd import Variable
    if isinstance(o, Variable) or torch.is_tensor(o):
        return o.cuda()
    return o


def as_cuda(obj):
    """
    Convert obj as_as_as_cuda.

    Args:
        obj: (todo): write your description
    """
    return stmap(_as_cuda, obj)


def _as_detached(o, clone=False):
    """
    Convert o as a tensor.

    Args:
        o: (todo): write your description
        clone: (str): write your description
    """
    from torch.autograd import Variable
    if isinstance(o, Variable) or torch.is_tensor(o):
        if clone:
            return o.clone().detach()
        return o.detach()
    return o


def as_detached(obj, clone=False):
    """
    Convert obj async object asynchronously.

    Args:
        obj: (todo): write your description
        clone: (todo): write your description
    """
    return stmap(functools.partial(_as_detached, clone=clone), obj)

