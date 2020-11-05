#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : custom_optimizer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['CustomizedOptimizer']


class CustomizedOptimizer(object):
    @property
    def state(self):
        """
        Set the state.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()

    @property
    def param_groups(self):
        """
        Return a list of parameter groups.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()

    def state_dict(self):
        """
        Set the state dict.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()

    def load_state_dict(self, state_dict):
        """
        Loads the state dictionary.

        Args:
            self: (todo): write your description
            state_dict: (dict): write your description
        """
        raise NotImplementedError()

    def zero_grad(self):
        """
        Return the gradient of the gradients.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()

    def step(self, closure=None):
        """
        Perform a step.

        Args:
            self: (todo): write your description
            closure: (callable): write your description
        """
        raise NotImplementedError()
