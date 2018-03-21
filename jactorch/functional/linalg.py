# -*- coding: utf-8 -*-
# File   : linalg.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/02/2018
# 
# This file is part of Jacinle.

__all__ = ['normalize']


def normalize(a, p=2, dim=-1, eps=1e-8):
    return a / a.norm(p, dim=dim, keepdim=True).clamp(min=eps)
