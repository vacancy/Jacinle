# -*- coding: utf-8 -*-
# File   : losses.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 25/01/2018
# 
# This file is part of Jacinle.

import torch
import torch.nn as nn
import torch.nn.functional as F

from jacinle.utils.enum import JacEnum
from jactorch.functional.indexing import one_hot, index_one_hot
from jactorch.graph.variable import var_with
from jactorch.functional.linalg import normalize
from jactorch.functional.mask import masked_average

__all__ = [
    'LossAverageMethod', 'AverageLoss',
    'CrossEntropyLossWithLogits', 'CrossEntropyLoss', 'MSELoss',
    'CrossEntropyLossWithProbs', 
    'SmoothL1Loss',
    'CompatibleCrossEntropyLossWithProbs', 'CompatibleMSEProbabilityLoss',
    'CosineLoss',
    'weighted_loss'
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


class CrossEntropyLossWithLogits(AverageLoss):
    def __init__(self, dim=-1, average=LossAverageMethod.VALID):
        super().__init__(average)
        self.dim = dim

    def forward(self, logits, target, mask=None):
        log_prob = F.log_softmax(logits, dim=self.dim)
        neg_xent = index_one_hot(log_prob, self.dim, target)
        return -self._average(neg_xent, mask)


CrossEntropyLoss = CrossEntropyLossWithLogits  # Typical PyTorch naming.


class MSELoss(AverageLoss):
    def __init__(self, average):
        super().__init__(average)

    def forward(self, output, target, mask=None):
        diff_sqr = 0.5 * ((output - target) ** 2)
        return self._average(diff_sqr, mask)


class CrossEntropyLossWithProbs(AverageLoss):
    _eps = 1e-8

    def __init__(self, dim=-1, average=LossAverageMethod.VALID):
        super().__init__(average)
        self.dim = dim

    def forward(self, probs, target, mask=None):
        log_prob = torch.log(probs.clamp(min=self._eps))
        neg_xent = index_one_hot(log_prob, self.dim, target)
        return -self._average(neg_xent, mask)


class SmoothL1Loss(AverageLoss):
    def __init__(self, sigma=3.0, average=LossAverageMethod.VALID):
        super().__init__(average)
        self.sigma2 = sigma * sigma

    def forward(self, input, target, sidechain=None):
        x = input - target
        a = (x >= 1.0 / self.sigma2).float()
        b = (x <= -1.0 / self.sigma2).float()
        loss = a * (x - 0.5 / self.sigma2) + b * (-x - 0.5 / self.sigma2) + (1 - a - b) * 0.5 * x * x * self.sigma2
        loss = loss.sum(dim=1)

        mask = None
        if sidechain is not None:
            mask = (sidechain > 0).float()
        return self._average(loss, mask)


class CompatibleCrossEntropyLossWithProbs(CrossEntropyLossWithProbs):
    def __init__(self, dim=-1, weight=None, ignore_index=None):
        super().__init__(dim, average=LossAverageMethod.NONE)
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
        loss = 0.5 * ((probs - target_onehot) ** 2.).sum(dim=1)
        return weighted_loss(loss, target, self.weight, self.ignore_index)


class CosineLoss(AverageLoss):
    def forward(self, pred, label, mask=None):
        input1 = normalize(pred, eps=1e-6)
        input2 = normalize(label, eps=1e-6)
        loss = 1 - (input1 * input2).sum(dim=-1)
        return self._average(loss, mask)


def weighted_loss(loss, target, weight, ignore_index):
    if weight is not None:
        weight = var_with(weight, target)
        weight = weight[target]
    else:
        weight = 1
    if ignore_index is not None:
        weight *= (target.ne(ignore_index).float())

    if type(weight) is int and weight == 1:
        return loss.mean()
    else:
        return masked_average(loss, weight)
