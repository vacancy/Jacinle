#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimizer_group.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/16/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from .custom_optimizer import CustomizedOptimizer

__all__ = ['OptimizerGroup']


class OptimizerGroup(CustomizedOptimizer):
    def __init__(self, **optimizers):
        self.optimizers = optimizers

    def __getattr__(self, item):
        return self.optimizers[item]

    def __getitem__(self, item):
        return self.optimizers[item]

    def state_dict(self):
        return {
            name: opt.state_dict()
            for name, opt in self.optimizers.items()
        }

    def load_state_dict(self, state_dict):
        for name, opt in state_dict.items():
            if name in self.optimizers:
                opt.load_state_dict(opt)

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for opt in self.optimizers:
            opt.step()

        return loss

