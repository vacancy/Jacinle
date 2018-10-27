#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : losses.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/25/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.nn as nn

from jacinle.utils.enum import JacEnum
from jactorch.functional.indexing import one_hot

from . import functional as F
from .functional import weighted_loss

__all__ = [
    'LossAverageMethod', 'AverageLoss',
    'BinaryCrossEntropyLossWithProbs', 'PNBalancedBinaryCrossEntropyLossWithProbs',
    'CrossEntropyLossWithLogits', 'CrossEntropyLoss', 'MSELoss',
    'CrossEntropyLossWithProbs',
    'SmoothL1Loss',
    'CompatibleCrossEntropyLossWithProbs', 'CompatibleMSEProbabilityLoss',
    'CosineLoss',
]


class LossAverageMethod(JacEnum):
    NONE = 'none'
    ALL = 'all'
    VALID = 'valid'


class AverageLoss(nn.Module):
    def __init__(self, average='valid'):
        super().__init__()
        self.average_method = LossAverageMethod.from_string(average)

    def _average(self, loss, mask):
        if self.average_method is not LossAverageMethod.NONE:
            if mask is not None:
                loss = loss * mask

                if self.average_method is LossAverageMethod.ALL:
                    loss = loss.mean()
                elif self.average_method is LossAverageMethod.VALID:
                    loss = loss.sum() / mask.sum().clamp(min=0.1)
                else:
                    raise ValueError('Unknown average method: {}.'.format(self.average_method))
            else:
                loss = loss.mean()
        return loss


class BinaryCrossEntropyLossWithProbs(AverageLoss):
    def forward(self, logits, target, mask=None):
        loss = F.binary_cross_entropy_with_probs(logits, target)
        return self._average(loss, mask)



class PNBalancedBinaryCrossEntropyLossWithProbs(nn.Module):
    def forward(self, probs, target, mask=None):
        return F.pn_balanced_binary_cross_entropy_with_probs(probs, target, mask)


class CrossEntropyLossWithLogits(AverageLoss):
    def __init__(self, dim=-1, average='valid'):
        super().__init__(average)
        self.dim = dim

    def forward(self, logits, target, mask=None):
        loss = F.cross_entropy_with_logits(logits, target, self.dim)
        return self._average(loss, mask)


CrossEntropyLoss = CrossEntropyLossWithLogits  # Typical PyTorch naming.


class MSELoss(AverageLoss):
    def __init__(self, average='valid'):
        super().__init__(average)

    def forward(self, output, target, mask=None):
        loss = F.l2_loss(output, target)
        return self._average(loss, mask)


class CrossEntropyLossWithProbs(AverageLoss):
    _eps = 1e-8

    def __init__(self, dim=-1, average='valid'):
        super().__init__(average)
        self.dim = dim

    def forward(self, probs, target, mask=None):
        loss = F.cross_entropy_with_probs(probs, target, self.dim, self._eps)
        return -self._average(loss, mask)


class SmoothL1Loss(AverageLoss):
    def __init__(self, sigma=3.0, average='valid'):
        super().__init__(average)
        self.sigma = sigma

    def forward(self, output, target, sidechain=None):
        loss = F.smooth_l1(output, target, self.sigma)
        loss = loss.sum(dim=-1)

        mask = None
        if sidechain is not None:
            mask = (sidechain > 0).float()
        return self._average(loss, mask)


class CompatibleCrossEntropyLossWithProbs(CrossEntropyLossWithProbs):
    def __init__(self, dim=-1, weight=None, ignore_index=None):
        super().__init__(dim, average='none')
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, probs, target, mask=None):
        assert mask is None
        loss = super().forward(probs, target)
        return weighted_loss(loss, target, self.weight, self.ignore_index)


class CompatibleMSEProbabilityLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, probs, target):
        target_onehot = one_hot(target, probs.size(1))
        loss = F.l2_loss(probs, target_onehot)
        return weighted_loss(loss, target, self.weight, self.ignore_index)


class CosineLoss(AverageLoss):
    def forward(self, output, target, mask=None):
        loss = F.cosine_loss(output, target)
        return self._average(loss, mask)
