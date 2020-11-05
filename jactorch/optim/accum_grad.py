#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : accum_grad.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['AccumGrad']

from .custom_optimizer import CustomizedOptimizer


class AccumGrad(CustomizedOptimizer):
    def __init__(self, base_optimizer, nr_acc):
        """
        Initialize the optimizer.

        Args:
            self: (todo): write your description
            base_optimizer: (todo): write your description
            nr_acc: (todo): write your description
        """
        self._base_optimizer = base_optimizer
        self._nr_acc = nr_acc
        self._current = 0

    @property
    def state(self):
        """
        : return : return state of the optimizer.

        Args:
            self: (todo): write your description
        """
        return self._base_optimizer.state

    @property
    def param_groups(self):
        """
        Return a list of the param_groups of this parameter.

        Args:
            self: (todo): write your description
        """
        return self._base_optimizer.param_groups

    def state_dict(self):
        """
        Return the current state of the current state.

        Args:
            self: (todo): write your description
        """
        # TODO(Jiayuan Mao @ 05/08): use a separate method to store all grad_buffer.
        return {
            'base_optimizer': self._base_optimizer.state_dict(),
            'current': self._current
        }

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer from a dictionary.

        Args:
            self: (todo): write your description
            state_dict: (dict): write your description
        """
        self._current = state_dict['current']
        return self._base_optimizer.load_state_dict(state_dict['base_optimizer'])

    def zero_grad(self):
        """
        The number of the gradient

        Args:
            self: (todo): write your description
        """
        return self._base_optimizer.zero_grad()

    def step(self, closure=None):
        """
        Perform an optimizer.

        Args:
            self: (todo): write your description
            closure: (callable): write your description
        """
        loss = None
        if closure is not None:
            loss = closure()

        self._current += 1

        for group in self._base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self._base_optimizer.state[p]

                # NB(Jiayuan Mao @ 02/16): we guarantee that grad_buffer does not require grad.
                if 'grad_buffer' not in param_state:
                    buf = param_state['grad_buffer'] = d_p.clone()
                else:
                    buf = param_state['grad_buffer']
                    buf.add_(d_p)

                if self._current >= self._nr_acc:
                    buf.mul_(1. / self._current)
                    p.grad.data.copy_(buf)
                    buf.zero_()

        if self._current >= self._nr_acc:
            self._base_optimizer.step()
            self._current = 0

        return loss
