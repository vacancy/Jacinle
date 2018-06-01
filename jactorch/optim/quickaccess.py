#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : quickaccess.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/28/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.optim as optim
import jactorch.optim as jacoptim


def get_optimizer(optimizer, model, *args, **kwargs):
    if isinstance(optimizer, (optim.Optimizer, jacoptim.CustomizedOptimizer)):
        return optimizer

    if type(optimizer) is str:
        try:
            optimizer = getattr(optim, optimizer)
        except AttributeError:
            try:
                optimizer = getattr(jacoptim, optimizer)
            except AttributeError:
                raise ValueError('Unknown optimizer type: {}.'.format(optimizer))

    return optimizer(filter(lambda p: p.requires_grad, model.parameters()), *args, **kwargs)
