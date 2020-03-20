#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn.functional as F

from jactorch.functional.shape import add_dim_as_except

__all__ = ['masked_softmax', 'length_masked_softmax']


def masked_softmax(logits, mask=None, dim=-1, eps=1e-20, ninf=-1e4):
    if mask is not None:
        logits = logits * mask + ninf * (1 - mask)

    probs = F.softmax(logits, dim=dim)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(dim, keepdim=True)

    return probs


def length_masked_softmax(logits, lengths, dim=-1, ninf=-1e4):
    rng = torch.arange(logits.size(dim=dim), dtype=lengths.dtype, device=lengths.device)
    rng = add_dim_as_except(rng, logits, dim)
    lengths = lengths.unsqueeze(dim)
    mask = rng < lengths
    return masked_softmax(logits, mask, dim=dim, ninf=ninf)

