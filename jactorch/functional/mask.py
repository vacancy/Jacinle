#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mask.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/08/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import numpy as np

from .shape import add_dim_as_except

__all__ = ['mask_meshgrid', 'masked_average', 'length2mask', 'length_masked_reversed']


def mask_meshgrid(mask, target_dims=2):
    for i in range(target_dims - 1):
        f = mask.unsqueeze(-1)
        g = mask.unsqueeze(-2)
        mask = f * g

    return mask


def masked_average(tensor, mask, eps=1e-8):
    """
    Compute the average of the tensor while ignoring some masked elements.

    Args:
        tensor (Tensor): tensor to be averaged.
        mask (Tensor): a mask indicating the element-wise weight.
        eps (float): eps for numerical stability.

    Returns:
        FloatTensor: the average of the input tensor.

    """
    tensor = tensor.float()
    mask = mask.float()
    masked = tensor * mask
    return masked.sum() / mask.sum().clamp(min=eps)


def length2mask(lengths, max_length):
    rng = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    lengths = lengths.unsqueeze(-1)
    rng = add_dim_as_except(rng, lengths, -1)
    mask = rng < lengths
    return mask.float()


def length_masked_reversed(tensor, lengths, dim=1):
    """
    Reverses sequences according to their lengths.

    Args:
        tensor (Tensor): padded batch of variable length sequences.
        lengths (LongTensor): list of sequence lengths

    Returns:
        Tensor: A Variable with the same size as tensor, but with each sequence
        reversed according to its length.

    """

    assert dim == 1

    if tensor.size(0) != len(lengths):
        raise ValueError('tensor incompatible with lengths.')
    reversed_indices = np.repeat(np.arange(tensor.size(1))[np.newaxis], inputs.size(0), 0)
    for i, length in enumerate(lengths.cpu().numpy().tolist()):
        if length > 0:
            reversed_indices[i, :length] = reversed_indices[i, length-1::-1]
    reversed_indices = torch.tensor(reversed_indices, dtype=torch.long, device=tensor.device)
    reversed_inputs = torch.gather(tensor, dim, reversed_indices)
    return reversed_inputs

