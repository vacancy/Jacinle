#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : shape.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/25/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Tensor shape utilities."""

import collections
from typing import Union, Tuple, List

import torch

__all__ = [
    'flatten', 'flatten2',
    'concat_shape',
    'broadcast', 'add_dim', 'add_dim_as_except', 'broadcast_as_except', 'move_dim',
    'repeat', 'repeat_times', 'force_view'
]


def flatten(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten the tensor."""
    return tensor.view(-1)


def flatten2(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten the tensor while keep the first (batch) dimension."""
    return tensor.view(tensor.size(0), -1)


def concat_shape(*shapes: Union[torch.Size, Tuple[int, ...], List[int], int]) -> Tuple[int, ...]:
    """Concatenate shapes into a tuple. The values can be either torch.Size, tuple, list, or int."""
    output = []
    for s in shapes:
        if isinstance(s, collections.Sequence):
            output.extend(s)
        else:
            output.append(int(s))
    return tuple(output)


def broadcast(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Broadcast a specific dim for `size` times. Originally the dim size must be 1.

    Example:

        >>> broadcast(torch.tensor([1, 2, 3]), 0, 2)
        tensor([[1, 2, 3],
                [1, 2, 3]])

    Args:
        tensor: the tensor to be broadcasted.
        dim: the dimension to be broadcasted.
        size: the size of the target dimension.

    Returns:
        the broadcasted tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    assert tensor.size(dim) == 1
    shape = tensor.size()
    return tensor.expand(concat_shape(shape[:dim], size, shape[dim + 1:]))


def add_dim(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Add a dimension at `dim` with size `size`.

    Example:

        >>> add_dim(torch.tensor([1, 2, 3]), 0, 2)
        tensor([[1, 2, 3],
                [1, 2, 3]])

    Args:
        tensor: the tensor to be broadcasted.
        dim: the new dimension to be added.
        size: the size of the target dimension.

    Returns:
        the broadcasted tensor.
    """
    return broadcast(tensor.unsqueeze(dim), dim, size)


def add_dim_as_except(tensor: torch.Tensor, target: torch.Tensor, *excepts: int) -> torch.Tensor:
    """Add dimension for the input tensor so that

    - It has the same number of dimensions as target.
    - The original axes of the tensor are ordered in `excepts`.

    Basically, it will "match" the shape of the target tensor, and align the axes in the input tensor
    with ``excepts`` in the target tensor.

    Note:

        The list excepts must be in ascending order.

    See Also:

        :func:`broadcast_as_except`, which adds AND expands dimension.

    Example:

        >>> add_dim_as_except(torch.rand(3, 4), torch.rand(2, 3, 4, 5), 1, 1).size()
        torch.Size([1, 3, 4, 1])

    Args:
        tensor: the tensor to be broadcasted.
        target: the target tensor.
        excepts: the dimensions to be kept.

    Returns:
        the broadcasted tensor.
    """
    assert len(excepts) == tensor.dim()
    tensor = tensor.clone()
    excepts = [e + target.dim() if e < 0 else e for e in excepts]
    for i in range(target.dim()):
        if i not in excepts:
            tensor.unsqueeze_(i)
    return tensor


def broadcast_as_except(tensor: torch.Tensor, target: torch.Tensor, *excepts: int) -> torch.Tensor:
    """Add AND expand dimension for the input tensor so that

        - It has the same number of dimensions as target.
        - The original axes of the tensor are ordered in `excepts`.
        - The original axes of the tensor are expanded to the size of the corresponding axes in target.

    After this function, the input tensor will have the same shape as the target tensor.

    Note:

        The list excepts must be in ascending order.

    See Also:

        :func:`add_dim_as_except`, which only adds dimension (without expanding).

    Example:

        >>> broadcast_as_except(torch.rand(3, 4), torch.rand(2, 3, 4, 5), 1, 1).size()
        torch.Size([2, 3, 4, 5])

    Args:
        tensor: the tensor to be broadcasted.
        target: the target tensor.
        excepts: the dimensions to be kept.

    Returns:
        the broadcasted tensor.
    """
    assert len(excepts) == tensor.dim()
    tensor_shape = tensor.size()
    target_shape = target.size()
    tensor = tensor.clone()
    excepts = [e + target.dim() if e < 0 else e for e in excepts]
    target_size = list()
    for i in range(target.dim()):
        if i not in excepts:
            target_size.append(target_shape[i])
            tensor.unsqueeze_(i)
        else:
            target_size.append(tensor_shape[excepts.index(i)])
    return tensor.expand(target_size)


def move_dim(tensor: torch.Tensor, dim: int, dest: int) -> torch.Tensor:
    """Move a specific dimension to a designated dimension."""
    if dest < 0:
        # CAUTION:: cannot rely on list.insert, because insert(['a', 'b', 'c'], -1, 'd') => 'abdc'
        dest += tensor.dim()
    dims = list(range(tensor.dim()))
    dims.pop(dim)
    dims.insert(dest, dim)
    return tensor.permute(dims)


def repeat(tensor: torch.Tensor, dim: int, count: int) -> torch.Tensor:
    """Repeat a specific dimension for `count` times.

    Example:

        >>> repeat(torch.tensor([1, 2, 3]), 0, 2)
        tensor([1, 2, 3, 1, 2, 3])

    See Also:

        :func:`repeat_times`, which repeats each element along the dimension.

    Args:
        tensor: the tensor to be repeated.
        dim: the dimension to be repeated.
        count: the number of times to repeat.

    Returns:
        the repeated tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    tensor_shape = tensor.size()
    value = broadcast(tensor.unsqueeze(dim + 1), dim + 1, count)
    return force_view(value, concat_shape(tensor_shape[:dim], -1, tensor_shape[dim + 1:]))


def repeat_times(tensor: torch.Tensor, dim: int, repeats: int) -> torch.Tensor:
    """Repeat each element along a specific dimension for `repeats` times.

    Example:

        >>> repeat_times(torch.tensor([1, 2, 3]), 0, 2)
        tensor([1, 1, 2, 2, 3, 3])

    See Also:

        :func:`repeat`, which repeats the whole dimension for `repeats` times.

    Args:
        tensor: the tensor to be repeated.
        dim: the dimension to be repeated.
        repeats: the number of times to repeat each element.

    Returns:
        the repeated tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    repeats = repeats.data.cpu().numpy()
    outputs = []
    for i in range(tensor.size(dim)):
        outputs.append(broadcast(tensor.narrow(dim, i, 1), dim, int(repeats[i])))
    return torch.cat(outputs, dim=dim)


def force_view(tensor: torch.Tensor, *shapes: int) -> torch.Tensor:
    """Do a view with optional contiguous copy. DEPRECATED. Use tensor.reshape instead."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.view(*shapes)

