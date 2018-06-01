#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : linalg.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['normalize']


def normalize(a, p=2, dim=-1, eps=1e-8):
    return a / a.norm(p, dim=dim, keepdim=True).clamp(min=eps)

