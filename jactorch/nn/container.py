# -*- coding: utf-8 -*-
# File   : container.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/04/2018
#
# This file is part of Jacinle.

import torch.nn as nn

__all__ = ['SequentialN']


class SequentialN(nn.Sequential):
    def forward(self, *inputs, return_all=False):
        all_values = [inputs]
        for module in self._modules.values():
            inputs = module(*inputs)
            all_values.append(inputs)
        if return_all:
            return inputs, all_values
        return inputs
