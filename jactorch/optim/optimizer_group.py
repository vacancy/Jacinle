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
        """
        Initialize the optimizer.

        Args:
            self: (todo): write your description
            optimizers: (todo): write your description
        """
        self.optimizers = optimizers

    def __getattr__(self, item):
        """
        Return the value of item.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return self.optimizers[item]

    def __getitem__(self, item):
        """
        Returns the item from the given item.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return self.optimizers[item]

    def state_dict(self):
        """
        Returns a dictionary of all variables.

        Args:
            self: (todo): write your description
        """
        return {
            name: opt.state_dict()
            for name, opt in self.optimizers.items()
        }

    def load_state_dict(self, state_dict):
        """
        Recursively load_dict.

        Args:
            self: (todo): write your description
            state_dict: (dict): write your description
        """
        for name, opt in state_dict.items():
            if name in self.optimizers:
                opt.load_state_dict(opt)

    def zero_grad(self):
        """
        Calculate the gradient of the optimizer.

        Args:
            self: (todo): write your description
        """
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self, closure=None):
        """
        Step optimizer.

        Args:
            self: (todo): write your description
            closure: (callable): write your description
        """
        loss = None
        if closure is not None:
            loss = closure()

        for opt in self.optimizers:
            opt.step()

        return loss

