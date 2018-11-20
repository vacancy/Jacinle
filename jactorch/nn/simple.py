#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simple.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/13/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.nn as nn

__all__ = ['Identity']


class Identity(nn.Module):
    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args

