#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simple.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/13/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

__all__ = ['Identity', 'TorchApplyRecorderMixin']


class Identity(nn.Module):
    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args


class TorchApplyRecorderMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._apply_recorder_indicator = nn.Parameter(
            torch.tensor(0, dtype=torch.float32, device=torch.device('cpu'))
        )
        self._apply_recorder_indicator.requires_grad = False

    @property
    def dtype(self):
        return self._apply_recorder_indicator.dtype

    @property
    def device(self):
        return self._apply_recorder_indicator.device
