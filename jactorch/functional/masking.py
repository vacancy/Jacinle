#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mask.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/08/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Masking-related functions."""

import torch
import torch.nn.functional as F
import numpy as np

from .shape import add_dim_as_except

__all__ = [
    'mask_meshgrid', 'masked_average', 'length2mask', 'length_masked_reversed',
    'masked_softmax', 'length_masked_softmax'
]


def mask_meshgrid(mask: torch.Tensor, target_dims: int = 2) -> torch.Tensor:
    """Create an N-dimensional meshgrid-like mask, where ``output[i, j, k, ...] = mask[i] * mask[j] * mask[k] * ...``.

    Args:
        mask: the original mask. Batch dimensions are supported, but the mask dimension is assumed to be the last one.
        target_dims: the number of target dimensions of the output mask.

    Returns:
        a mask with shape ``mask.shape + (target_dims - mask.dim())``.
    """
    for i in range(target_dims - 1):
        f = mask.unsqueeze(-1)
        g = mask.unsqueeze(-2)
        mask = f * g

    return mask


def masked_average(tensor: torch.Tensor, mask: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """Compute the average of the tensor while ignoring some masked elements.

    Args:
        tensor: tensor to be averaged.
        mask: a mask indicating the element-wise weight.
        eps: eps for numerical stability.

    Returns:
        the average of the input tensor.
    """
    tensor = tensor.float()
    mask = mask.float()
    masked = tensor * mask
    return masked.sum() / mask.sum().clamp(min=eps)


def length2mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """Convert a length vector to a mask.

    Args:
        lengths: a vector of length. Batch dimensions are supported, but the length dimension is assumed to be the last one.
        max_length: the maximum length of the mask.

    Returns:
        a mask with shape ``lengths.shape + (max_length,)``.
    """
    rng = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    lengths = lengths.unsqueeze(-1)
    rng = add_dim_as_except(rng, lengths, -1)
    mask = rng < lengths
    return mask.float()


def length_masked_reversed(tensor: torch.Tensor, lengths: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Reverse a padded sequence tensor along the given dimension.

    Args:
        tensor: padded batch of variable length sequences.
        lengths: list of sequence lengths
        dim: dimension along which to reverse sequences. Currently only supports dim=1.

    Returns:
        A tensor with the same size as the input, but with each sequence reversed.
    """
    assert dim == 1

    if tensor.size(0) != len(lengths):
        raise ValueError('tensor incompatible with lengths.')
    reversed_indices = np.repeat(np.arange(tensor.size(1))[np.newaxis], tensor.size(0), 0)
    for i, length in enumerate(lengths.cpu().numpy().tolist()):
        if length > 0:
            reversed_indices[i, :length] = reversed_indices[i, length-1::-1]
    reversed_indices = torch.tensor(reversed_indices, dtype=torch.long, device=tensor.device)
    reversed_inputs = torch.gather(tensor, dim, reversed_indices)
    return reversed_inputs


def masked_softmax(logits, mask=None, dim=-1, eps=1e-20, ninf=-1e4):
    """Compute the softmax of the tensor while ignoring some masked elements.
    When all elements are masked, the result is a uniform distribution.

    Args:
        logits: tensor to be softmaxed.
        mask: a mask indicating the element-wise weight.
        dim: the dimension to be softmaxed.
        eps: eps for numerical stability.
        ninf: the value to be used for masked elements.

    Returns:
        the softmax of the input tensor.
    """
    if mask is not None:
        logits = logits.masked_fill(~mask, ninf)

    probs = F.softmax(logits, dim=dim)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(dim, keepdim=True)

    return probs


def length_masked_softmax(logits, lengths, dim=-1, ninf=-1e4):
    """Compute the softmax of the tensor while ignoring some masked elements.
    Unlike :func:`masked_softmax`, this function uses the lengths to compute the mask.
    When all elements are masked, the result is a uniform distribution.

    Args:
        logits: tensor to be softmaxed.
        lengths: a vector of length. Batch dimensions are supported, but the length dimension is assumed to be the last one.
        dim: the dimension to be softmaxed.
        ninf: the value to be used for masked elements.

    Returns:
        the softmax of the input tensor.
    """
    rng = torch.arange(logits.size(dim=dim), dtype=lengths.dtype, device=lengths.device)
    rng = add_dim_as_except(rng, logits, dim)
    lengths = lengths.unsqueeze(dim)
    mask = rng < lengths
    return masked_softmax(logits, mask, dim=dim, ninf=ninf)

