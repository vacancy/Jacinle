#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/24/2018
#
# Distributed under terms of the MIT license.

import torch
import torch.nn.functional as F

from jactorch.functional.shape import add_dim_as_except

__all__ = ['masked_softmax', 'length_masked_softmax', 'length2mask', 'mask_meshgrid']


def masked_softmax(logits, mask, dim=-1, ninf=-1e4):
    mask = mask.float()
    ninf = float(ninf)
    masked_logits = logits * mask + (1 - mask) * ninf
    return F.softmax(logits, dim=dim)


def length_masked_softmax(logits, lengths, dim=-1, ninf=-1e4):
    rng = torch.arange(logits.size(dim=dim), dtype=lengths.dtype, device=lengths.device)
    rng = add_dim_as_except(rng, logits, dim)
    lengths = lengths.unsqueeze(dim)
    mask = rng < lengths
    return masked_softmax(logits, mask, dim=dim, ninf=ninf)


def length2mask(lengths, max_length):
    rng = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    lengths = lengths.unsqueeze(-1)
    rng = add_dim_as_except(rng, lengths, -1)
    mask = rng < lengths
    return mask.float()


def mask_meshgrid(mask, target_dims=2):
    for i in range(target_dims - 1):
        f = mask.unsqueeze(-1)
        g = mask.unsqueeze(-2)
        mask = f * g

    return mask

