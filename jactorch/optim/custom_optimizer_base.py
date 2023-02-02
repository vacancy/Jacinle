#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : custom_optimizer_base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['CustomizedOptimizer']


class CustomizedOptimizer(object):
    """Base class for all customized optimizers."""

    @property
    def state(self):
        """The state of the optimizer."""
        raise NotImplementedError()

    @property
    def param_groups(self):
        """The parameter groups of the optimizer."""
        raise NotImplementedError()

    def state_dict(self):
        """A dictionary that contains the state of the optimizer."""
        raise NotImplementedError()

    def load_state_dict(self, state_dict):
        """Load the state of the optimizer from a dictionary."""
        raise NotImplementedError()

    def zero_grad(self):
        """Clear the gradients of all optimized parameters."""
        raise NotImplementedError()

    def step(self, closure=None):
        """Performs a single optimization step."""
        raise NotImplementedError()
