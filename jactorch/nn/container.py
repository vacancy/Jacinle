#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : container.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/09/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.nn as nn

__all__ = ['SequentialN', 'AutoResetParametersMixin']


class SequentialN(nn.Sequential):
    def forward(self, *inputs, return_all=False):
        all_values = [inputs]
        for module in self._modules.values():
            inputs = module(*inputs)
            all_values.append(inputs)
        if return_all:
            return inputs, all_values
        return inputs


class AutoResetParametersMixin(object):
    def reset_parameters(self):
        for module in self.modules():
            if id(module) != id(self) and hasattr(module, 'reset_parameters'):
                module.reset_parameters()

