#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : indexing.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Tensor indexing utils."""

import torch

from jacinle.utils.numeric import prod
from jacinle.utils.vendor import has_vendor, requires_vendors
from jactorch.utils.grad import no_grad_func
from .shape import concat_shape, add_dim_as_except

__all__ = [
    'reversed',
    'one_hot', 'one_hot_nd', 'one_hot_dim',
    'inverse_permutation',
    'index_one_hot', 'set_index_one_hot_', 'index_one_hot_ellipsis',
    'leftmost_nonzero', 'rightmost_nonzero',
    'index_nonzero',
    'batched_index_select',
    'batch', 'patch_torch_index',
    'batched_index_int', 'batched_index_slice', 'batched_index_vector_dim', 'batched_index_vectors',
    'tindex', 'findex', 'vindex', 'oindex',
    'btindex', 'bfindex', 'bvindex', 'boindex'
]


def reversed(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Reverse a tensor along the given dimension. For example, if `dim=0`, it is equivalent to
    the python notation: `x[::-1]`.

    Args:
        x (torch.Tensor): input.
        dim (int): the dimension to be reversed.

    Returns:
        torch.Tensor: of same shape as `x`, but with the dimension `dim` reversed.

    """
    # https://github.com/pytorch/pytorch/issues/229#issuecomment-350041662
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    inds = torch.arange(x.size(1) - 1, -1, -1, dtype=torch.long, device=x.device)
    x = x.view(x.size(0), x.size(1), -1)[:, inds, :]
    return x.view(xsize)


@no_grad_func
def one_hot(index: torch.Tensor, nr_classes: int) -> torch.Tensor:
    """
    Convert a list of class labels into one-hot representation.

    .. note::
        This function support only one-dimensional input. For high dimensional inputs, use `one_hot_nd`.

    Args:
        index (torch.Tensor): shape `(N, )`, input class labels.
        nr_classes (int): number of total classes.

    Returns:
        torch.Tensor: shape `(N, nr_classes)`, one-hot representation of the class labels.
    """
    assert index.dim() == 1
    mask = torch.zeros(index.size(0), nr_classes, dtype=torch.float32, device=index.device)
    ones = torch.ones(index.size(0), 1, dtype=torch.float32, device=index.device)
    ret = mask.scatter_(1, index.unsqueeze(1), ones)
    return ret


@no_grad_func
def one_hot_nd(index: torch.Tensor, nr_classes: int) -> torch.Tensor:
    """
    Convert a tensor of class labels into one-hot representation.

    Args:
        index (torch.Tensor): input class labels.
        nr_classes (int): number of total classes.

    Returns:
        torch.Tensor: one-hot representation of the class labels, the label dimension is assumed to be the last one.
    """
    index_size = index.size()
    return one_hot(index.reshape(-1), nr_classes).view(index_size + (nr_classes, ))


@no_grad_func
def one_hot_dim(index: torch.Tensor, nr_classes: int, dim: int) -> torch.Tensor:
    """
    Convert a tensor of class labels into one-hot representation by adding a new dimension indexed at `dim`.

    Args:
        index (torch.Tensor): input class labels.
        nr_classes (int): number of total classes.
        dim (int): dimension of the class label.

    Returns:
        torch.Tensor: one-hot representation of the class labels.
    """
    return one_hot_nd(index, nr_classes).transpose(-1, dim)


@no_grad_func
def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    """
    Inverse a permutation.

    .. warning::
        This function does not check the validness of the input. That is, if the input is not a permutation, this
        function may generate arbitrary output.

    Args:
        perm (torch.Tensor): shape `(N, )` representing a permutation of 0 ~ N - 1.

    Returns:
        torch.Tensor: the inverse permutation, which satisfies: `inv[perm[x]] = x`.
    """
    assert perm.dim() == 1
    length = perm.size(0)
    inv = torch.zeros(length, dtype=torch.long, device=perm.device)
    inv.scatter_(0, perm, torch.arange(0, length, dtype=torch.long, device=perm.device))
    return inv.long()


def index_one_hot(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """`tensor[:, :, index, :]`

    Args:
        tensor (torch.Tensor): input.
        dim (int) the dimension.
        index: (torch.Tensor): the tensor containing the indices along the `dim` dimension.

    Returns:
        torch.Tensor: `tensor[:, :, index, :, :]`.
    """
    return tensor.gather(dim, index.unsqueeze(dim)).squeeze(dim)


def set_index_one_hot_(tensor: torch.Tensor, dim: int, index: torch.Tensor, value: torch.Tensor) -> None:
    """`tensor[:, :, index, :, :] = value`.

    Args:
        tensor (torch.Tensor): input.
        dim (int) the dimension.
        index: (torch.Tensor): the tensor containing the indices along the `dim` dimension.
        value (torch.Tensor): the value to be set.
    """
    if not isinstance(value, (int, float)):
        value = value.unsqueeze(dim)
    tensor.scatter_(dim, index.unsqueeze(dim), value)


def index_one_hot_ellipsis(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """`tensor[:, :, index, ...]`.

    Args:
        tensor (torch.Tensor): input.
        dim (int) the dimension.
        index: (torch.Tensor): the tensor containing the indices along the `dim` dimension.

    Returns:
        torch.Tensor: `tensor[:, :, index, ...]`.
    """
    tensor_shape = tensor.size()
    tensor = tensor.view(prod(tensor_shape[:dim]), tensor_shape[dim], prod(tensor_shape[dim+1:]))
    assert tensor.size(0) == index.size(0)
    index = index.unsqueeze(-1).unsqueeze(-1)
    index = index.expand(tensor.size(0), 1, tensor.size(2))
    tensor = tensor.gather(1, index)
    return tensor.view(tensor_shape[:dim] + tensor_shape[dim+1:])


def leftmost_nonzero(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Return the smallest nonzero index along the `dim` axis. The tensor should be binary.

    Args:
        tensor (torch.Tensor): input.
        dim (int): the dimension.

    Returns:
        torch.Tensor: the smallest nonzero index along the `dim` axis.
    """
    indices = add_dim_as_except(
        torch.arange(tensor.size(dim) - 1, -1, -1, dtype=torch.int64, device=tensor.device),
        tensor, dim
    )
    return (tensor.to(torch.int64) * tensor.size(dim) + indices).argmax(dim=dim)


def rightmost_nonzero(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Return the smallest nonzero index along the `dim` axis. The tensor should be binary.

    Args:
        tensor (torch.Tensor): input.
        dim (int): the dimension.

    Returns:
        torch.Tensor : the smallest nonzero index along the `dim` axis.
    """
    indices = add_dim_as_except(
        torch.arange(tensor.size(dim), dtype=torch.int64, device=tensor.device),
        tensor, dim
    )
    return (tensor.to(torch.int64) * tensor.size(dim) + indices).argmax(dim=dim)


def index_nonzero(tensor, mask):
    """Iteratively generates the values of `tensor` where `mask` is nonzero. When `mask` is a 1D tensor,
    this function is equivalent to:

    .. code-block:: python

        for i in range(mask.size(0)):
            if mask[i]:
                yield tensor[i]

    Args:
        tensor (torch.Tensor): input.
        mask (torch.Tensor): the mask.

    Yields:
        torch.Tensor: the values of `tensor` where `mask` is nonzero.
    """

    assert tensor.shape[:mask.dim()] == mask.shape
    if mask.dim() == 0:
        if mask.item() != 0:
            if tensor.dim() == 0:
                yield tensor
            else:
                yield tensor[0]
    else:
        yield from tensor[torch.not_equal(mask, 0)]


def batched_index_select(tensor: torch.Tensor, batched_indices: torch.Tensor) -> torch.Tensor:
    """Select elements from `tensor` according to `batched_indices`. The first dimension is assumed to be the batch dimension.
    This operation is equivalent to numpy: `tensor[np.arange(len(batched_indices)), batched_indices]`.

    Args:
        tensor (torch.Tensor): input.
        batched_indices (torch.Tensor): the indices to be selected.

    Returns:
        torch.Tensor: the selected elements.
    """
    assert batched_indices.dim() == 2

    batch_i = torch.arange(batched_indices.size(0)).to(batched_indices)
    batch_i = batch_i.unsqueeze(-1).expand_as(batched_indices)
    flattened_indices = batched_indices + batch_i * batched_indices.size(1)

    return (tensor
        .reshape(concat_shape(-1, tensor.size()[2:]))[flattened_indices.view(-1)]
        .reshape(concat_shape(batched_indices.size(), tensor.size()[2:]))
    )


if has_vendor('torch_index'):
    from torch_index import batch
    from torch_index import patch_torch as patch_torch_index
    from torch_index import tindex, findex, vindex, oindex
    from torch_index import btindex, bfindex, bvindex, boindex
    from torch_index.batched_functional import batched_index_int, batched_index_slice, batched_index_vector_dim, batched_index_vectors
else:
    from jacinle.utils.meta import make_dummy_func
    batch = slice(None, None, None)
    patch_torch_index = requires_vendors('torch_index')(make_dummy_func())
    tindex = requires_vendors('torch_index')(make_dummy_func())
    findex = requires_vendors('torch_index')(make_dummy_func())
    vindex = requires_vendors('torch_index')(make_dummy_func())
    oindex = requires_vendors('torch_index')(make_dummy_func())
    btindex = requires_vendors('torch_index')(make_dummy_func())
    bfindex = requires_vendors('torch_index')(make_dummy_func())
    bvindex = requires_vendors('torch_index')(make_dummy_func())
    boindex = requires_vendors('torch_index')(make_dummy_func())
    batched_index_int = requires_vendors('torch_index')(make_dummy_func())
    batched_index_slice = requires_vendors('torch_index')(make_dummy_func())
    batched_index_vector_dim = requires_vendors('torch_index')(make_dummy_func())
    batched_index_vectors = requires_vendors('torch_index')(make_dummy_func())


if has_vendor('einshape'):
    from einshape.src.pytorch.pytorch_ops import einshape
else:
    from jacinle.utils.meta import make_dummy_func
    einshape = requires_vendors('einshape')(make_dummy_func())

