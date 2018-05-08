#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : grad.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/08/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import functools

import torch

__all__ = ['no_grad_func']


def no_grad_func(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func
