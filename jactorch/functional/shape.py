#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : shape.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/25/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections
import torch

__all__ = ['flatten', 'flatten2', 'concat_shape', 'broadcast', 'add_dim', 'add_dim_as_except', 'repeat', 'repeat_times', 'force_view']


def flatten(tensor):
    return tensor.view(-1)


def flatten2(tensor):
    return tensor.view(tensor.size(0), -1)


def concat_shape(*shapes):
    output = []
    for s in shapes:
        if isinstance(s, collections.Sequence):
            output.extend(s)
        else:
            output.append(int(s))
    return tuple(output)


def broadcast(tensor, dim, size):
    if dim < 0:
        dim += tensor.dim()
    assert tensor.size(dim) == 1
    shape = tensor.size()
    return tensor.expand(concat_shape(shape[:dim], size, shape[dim+1:]))


def add_dim(tensor, dim, size):
    return broadcast(tensor.unsqueeze(dim), dim, size)


def add_dim_as_except(tensor, target, *excepts):
    assert len(excepts) == tensor.dim()
    tensor = tensor.clone()
    excepts = [e + target.dim() if e < 0 else e for e in excepts]
    for i in range(target.dim()):
        if i not in excepts:
            tensor.unsqueeze_(i)
    return tensor


def move_dim(tensor, dim, dest):
    dims = list(range(tensor.dim()))
    dims.pop(dim)
    dims.insert(dest, dim)
    return tensor.permute(dims)


def repeat(tensor, dim, count):
    if dim < 0:
        dim += tensor.dim()
    tensor_shape = tensor.size()
    value = broadcast(tensor.unsqueeze(dim + 1), dim + 1, count)
    return force_view(value, concat_shape(tensor_shape[:dim], -1, tensor_shape[dim + 1:]))


def repeat_times(tensor, dim, repeats):
    if dim < 0:
        dim += tensor.dim()
    repeats = repeats.data.cpu().numpy()
    outputs = []
    for i in range(tensor.size(dim)):
        outputs.append(broadcast(tensor.narrow(dim, i, 1), dim, int(repeats[i])))
    return torch.cat(outputs, dim=dim)


def force_view(tensor, *shapes):
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.view(*shapes)
