#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/03/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.utils.deprecated import deprecated
from jactorch.graph.parameter import mark_freezed as gmark_freezed, mark_unfreezed as gmark_unfreezed

__all__ = ['mark_freezed', 'mark_unfreezed', 'set_learning_rate', 'decay_learning_rate']


@deprecated
def mark_freezed(model):
    """DEPRECATED(Jiayuan Mao): jactorch.train.utils.mark_freezed has been deprecated and will be removed by 07/16/2022;
    please use jactorch.graph.parameter.mark_freezed instead."""
    return gmark_freezed(model)


@deprecated
def mark_unfreezed(model):
    """DEPRECATED(Jiayuan Mao): jactorch.train.utils.mark_unfreezed has been deprecated and will be removed by 07/16/2022;
    please use jactorch.graph.parameter.mark_unfreezed instead."""
    return gmark_unfreezed(model)


def set_learning_rate(optimizer, lr: float):
    """Set the learning rate of the optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def decay_learning_rate(optimizer, decay: float):
    """Decay the learning rate of the optimizer by a factor of decay."""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

