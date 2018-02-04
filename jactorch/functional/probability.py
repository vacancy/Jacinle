# -*- coding: utf-8 -*-
# File   : probability.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/02/2018
# 
# This file is part of Jacinle.


def normalize_prob(a, dim=-1):
    return a / a.sum(dim=dim, keepdim=True)
