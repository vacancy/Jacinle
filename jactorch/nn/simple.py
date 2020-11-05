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
        """
        Returns the forward. forward of args.

        Args:
            self: (todo): write your description
        """
        if len(args) == 1:
            return args[0]
        return args


class TorchApplyRecorderMixin(nn.Module):
    def __init__(self):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
        """
        super().__init__()
        self._apply_recorder_indicator = nn.Parameter(
            torch.tensor(0, dtype=torch.float32, device=torch.device('cpu'))
        )
        self._apply_recorder_indicator.requires_grad = False

    @property
    def dtype(self):
        """
        The dtype of this space.

        Args:
            self: (todo): write your description
        """
        return self._apply_recorder_indicator.dtype

    @property
    def device(self):
        """
        The device. device. device. device. device.

        Args:
            self: (todo): write your description
        """
        return self._apply_recorder_indicator.device
