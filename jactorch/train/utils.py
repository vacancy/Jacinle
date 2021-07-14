#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/03/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['mark_freezed', 'mark_unfreezed', 'set_learning_rate', 'decay_learning_rate']


def mark_freezed(model):
    model.eval()  # Turn off all BatchNorm / Dropout
    for p in model.parameters():
        p.requires_grad = False


def mark_unfreezed(model):
    model.train()  # Turn on all BatchNorm / Dropout
    for p in model.parameters():
        p.requires_grad = True


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def decay_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
