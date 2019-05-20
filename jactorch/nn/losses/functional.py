#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/01/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn.functional as F

from jactorch.functional import index_one_hot, masked_average, normalize

__all__ = [
    'weighted_loss',
    'pn_balanced_binary_cross_entropy_with_probs',
    'cross_entropy_with_logits', 'cross_entropy_with_probs',
    'l2_loss', 'smooth_l1', 'cosine_loss'
]


def weighted_loss(loss, target, weight, ignore_index):
    if weight is not None:
        weight = weight[target]
    else:
        weight = 1
    if ignore_index is not None:
        weight *= (target.ne(ignore_index).float())

    if type(weight) is int and weight == 1:
        return loss.mean()
    else:
        return masked_average(loss, weight)


def binary_cross_entropy_with_probs(probs, target, eps=1e-6):
    probs_1m = 1 - probs
    target_1m = 1 - target
    loss = -target * probs.clamp(min=eps).log() - target_1m * probs_1m.clamp(min=eps).log()

    return loss


def pn_balanced_binary_cross_entropy_with_probs(probs, target, mask=None, eps=1e-6):
    pos_mask = (target > 0.5).float()
    neg_mask = 1 - pos_mask

    if mask is not None:
        pos_mask *= mask
        neg_mask *= mask

    pos_count = pos_mask.sum()
    neg_count = neg_mask.sum()

    pos_mask1 = pos_mask / (pos_count * 2).clamp(min=eps)
    neg_mask1 = neg_mask / (neg_count * 2).clamp(min=eps)

    loss = binary_cross_entropy_with_probs(probs, target, eps)
    if mask is not None:
        loss = loss * mask
    return (loss * pos_mask1 + loss * neg_mask1).sum()


def cross_entropy_with_logits(logits, target, dim):
    log_prob = F.log_softmax(logits, dim)
    neg_xent = index_one_hot(log_prob, dim, target)
    return -neg_xent


def l2_loss(output, target):
    return 0.5 * ((output - target) ** 2)


def cross_entropy_with_probs(probs, target, dim=-1, eps=1e-8):
    log_prob = torch.log(probs.clamp(min=eps))
    neg_xent = index_one_hot(log_prob, dim, target)
    return -neg_xent


def smooth_l1(output, target, sigma):
    sigma2 = sigma * sigma
    x = output - target
    a = (x >= 1.0 / sigma2).float()
    b = (x <= -1.0 / sigma2).float()
    loss = a * (x - 0.5 / sigma2) + b * (-x - 0.5 / sigma2) + (1 - a - b) * 0.5 * x * x * sigma2
    return loss


def cosine_loss(output, target):
    input1 = normalize(output, eps=1e-6)
    input2 = normalize(target, eps=1e-6)
    loss = 1 - (input1 * input2).sum(dim=-1)
    return loss
