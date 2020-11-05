#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : _utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/07/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import functools
import torch
import torch.cuda as cuda


def auto_device(func):
    """
    Decorator to auto auto auto - auto - auto - auto - auto - auto - auto - auto - auto - auto - auto - auto -

    Args:
        func: (todo): write your description
    """
    @functools.wraps(func)
    def wrapped(tensor, *args, **kwargs):
        """
        Decorator to wrap tensor.

        Args:
            tensor: (todo): write your description
        """
        if tensor.device.type == 'cuda':
            with cuda.device(tensor.device):
                return func(tensor, *args, **kwargs)
        return func(tensor, *args, **kwargs)
    return wrapped

