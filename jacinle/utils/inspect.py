#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : inspect.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/05/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
Code inspection tools.
"""

__all__ = ['bind_args']


def bind_args(sig, *args, **kwargs):
    bounded = sig.bind(*args, **kwargs)
    bounded.apply_defaults()

    return bounded

