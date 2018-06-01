#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/03/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['mark_freezed', 'mark_unfreezed']


def mark_freezed(model):
    model.eval()  # Turn off all BatchNorm / Dropout
    for p in model.parameters():
        p.requires_grad = False


def mark_unfreezed(model):
    model.train()  # Turn on all BatchNorm / Dropout
    for p in model.parameters():
        p.requires_grad = True
