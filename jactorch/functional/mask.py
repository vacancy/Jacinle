#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mask.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/08/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['masked_average', 'length_masked_reversed']

import torch
import numpy as np


def masked_average(tensor, mask, eps=1e-8):
    tensor = tensor.float()
    mask = mask.float()
    masked = tensor * mask
    return masked.sum() / mask.sum().clamp(min=eps)


def length_masked_reversed(tensor, lengths, dim=1):
    """Reverses sequences according to their lengths.
    Arguments:
        tensor (torch.Tensor): padded batch of variable length sequences.
        lengths (torch.LongTensor): list of sequence lengths
    Returns:
        A Variable with the same size as tensor, but with each sequence
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

