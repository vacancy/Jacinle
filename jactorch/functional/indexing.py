# -*- coding: utf-8 -*-
# File   : indexing.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 24/01/2018
# 
# This file is part of Jacinle.

import torch

from jacinle.utils.numeric import prod
from jactorch.graph.variable import var_with, new_var_with


def reversed(x, dim=-1):
    # https://github.com/pytorch/pytorch/issues/229#issuecomment-350041662
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    inds = var_with(torch.arange(x.size(1) - 1, -1, -1).long(), x)
    x = x.view(x.size(0), x.size(1), -1)[:, inds, :]
    return x.view(xsize)


def one_hot(index, nr_classes):
    assert index.dim() == 1
    mask = new_var_with(index, index.size(0), nr_classes).fill_(0)
    ones = new_var_with(index, index.size(0), 1).fill_(1)
    ret = mask.scatter_(1, index.unsqueeze(-1), ones)
    return ret


def inverse_permutation(perm):
    length = perm.size(0)
    inv = var_with(perm.data.new(length).long().zero_(), perm)
    inv.scatter_(0, perm, var_with(torch.arange(length).long(), perm))
    return inv.long()


def index_one_hot(tensor, dim, index):
    tensor_shape = tensor.size()
    tensor = tensor.view(prod(tensor_shape[:dim]), tensor_shape[dim], prod(tensor_shape[dim+1:]))
    assert tensor.size(0) == index.size(0)
    index = index.unsqueeze(-1).unsqueeze(-1)
    index = index.expand(tensor.size(0), 1, tensor.size(2))
    tensor = tensor.gather(1, index)
    return tensor.view(tensor_shape[:dim] + tensor_shape[dim+1:])

