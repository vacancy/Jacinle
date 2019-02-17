#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : indexing.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch

from jacinle.utils.numeric import prod
from jactorch.utils.grad import no_grad_func
from .shape import concat_shape

__all__ = [
    'reversed',
    'one_hot', 'one_hot_nd', 'one_hot_dim',
    'inverse_permutation',
    'index_one_hot', 'set_index_one_hot_',
    'index_one_hot_ellipsis',
    'batch_index_select'
]


def reversed(x, dim=-1):
    """
    Reverse a tensor along the given dimension. For example, if `dim=0`, it is equivalent to
    the python notation: `x[::-1]`.

    Args:
        x (Tensor): input.
        dim: the dimension to be reversed.

    Returns:
        Tensor: of same shape as `x`, but with the dimension `dim` reversed.

    """
    # https://github.com/pytorch/pytorch/issues/229#issuecomment-350041662
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    inds = torch.arange(x.size(1) - 1, -1, -1, dtype=torch.long, device=x.device)
    x = x.view(x.size(0), x.size(1), -1)[:, inds, :]
    return x.view(xsize)


@no_grad_func
def one_hot(index, nr_classes):
    """
    Convert a list of class labels into one-hot representation.

    .. note::
        This function support only one-dimensional input. For high dimensional inputs, use `one_hot_nd`.

    Args:
        index (Tensor): shape `(N, )`, input class labels.
        nr_classes (int): number of total classes.

    Returns:
        Tensor: shape `(N, nr_classes)`, one-hot representation of the class labels.

    """
    assert index.dim() == 1
    mask = torch.zeros(index.size(0), nr_classes, dtype=torch.float32, device=index.device)
    ones = torch.ones(index.size(0), 1, dtype=torch.float32, device=index.device)
    ret = mask.scatter_(1, index.unsqueeze(1), ones)
    return ret


@no_grad_func
def one_hot_nd(index, nr_classes):
    """
    Convert a tensor of class labels into one-hot representation.

    Args:
        index (Tensor): input class labels.
        nr_classes (int): number of total classes.

    Returns:
        Tensor: one-hot representation of the class labels, the label dimension is assumed to be the last one.

    """
    index_size = index.size()
    return one_hot(index.view(-1), nr_classes).view(index_size + (nr_classes, ))


@no_grad_func
def one_hot_dim(index, nr_classes, dim):
    """
    Convert a tensor of class labels into one-hot representation by adding a new dimension indexed at `dim`.

    Args:
        index (Tensor): input class labels.
        nr_classes (int): number of total classes.
        dim (int): dimension of the class label.

    Returns:
        Tensor: one-hot representation of the class labels.

    """
    return one_hot_nd(index, nr_classes).transpose(-1, dim)


@no_grad_func
def inverse_permutation(perm):
    """
    Inverse a permutation.

    .. warning::
        This function does not check the validness of the input. That is, if the input is not a permutation, this
        function may generate arbitrary output.

    Args:
        perm (LongTensor): shape `(N, )` representing a permutation of 0 ~ N - 1.

    Returns:
        LongTensor: the inverse permutation, which satisfies: `inv[perm[x]] = x`.

    """
    assert perm.dim() == 1
    length = perm.size(0)
    inv = torch.zeros(length, dtype=torch.long, device=perm.device)
    inv.scatter_(0, perm, torch.arange(0, length, dtype=torch.long, device=perm.device))
    return inv.long()


def index_one_hot(tensor, dim, index):
    """
    Args:
        tensor (Tensor): input.
        dim (int) the dimension.
        index: (LongTensor): the tensor containing the indices along the `dim` dimension.

    Returns:
        Tensor: `tensor[:, :, index, :, :]`.

    """
    return tensor.gather(dim, index.unsqueeze(dim)).squeeze(dim)


def set_index_one_hot_(tensor, dim, index, value):
    """
    `tensor[:, :, index, :, :] = value`.

    Args:
        tensor (Tensor): input.
        dim (int) the dimension.
        index: (LongTensor): the tensor containing the indices along the `dim` dimension.

    """
    if not isinstance(value, (int, float)):
        value = value.unsqueeze(dim)
    tensor.scatter_(dim, index.unsqueeze(dim), value)


def index_one_hot_ellipsis(tensor, dim, index):
    """
    Args:
        tensor (Tensor): input.
        dim (int) the dimension.
        index: (LongTensor): the tensor containing the indices along the `dim` dimension.

    Returns:
        Tensor: `tensor[:, :, index, ...]`.

    """
    tensor_shape = tensor.size()
    tensor = tensor.view(prod(tensor_shape[:dim]), tensor_shape[dim], prod(tensor_shape[dim+1:]))
    assert tensor.size(0) == index.size(0)
    index = index.unsqueeze(-1).unsqueeze(-1)
    index = index.expand(tensor.size(0), 1, tensor.size(2))
    tensor = tensor.gather(1, index)
    return tensor.view(tensor_shape[:dim] + tensor_shape[dim+1:])


def batch_index_select(tensor, batched_indices):
    assert batched_indices.dim() == 2

    batch_i = torch.arange(batched_indices.size(0)).to(batched_indices)
    batch_i = batch_i.unsqueeze(-1).expand_as(batched_indices)
    flattened_indices = batched_indices + batch_i * batched_indices.size(1)

    return (tensor
        .reshape(concat_shape(-1, tensor.size()[2:]))[flattened_indices.view(-1)]
        .reshape(concat_shape(batched_indices.size(), tensor.size()[2:]))
    )

