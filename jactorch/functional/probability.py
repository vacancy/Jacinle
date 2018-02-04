# -*- coding: utf-8 -*-
# File   : probability.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/02/2018
# 
# This file is part of Jacinle.


def normalize_prob(a, dim=-1):
    return a / a.sum(dim=dim, keepdim=True)


def check_prob_normalization(p, atol=1e-5):
    tot = p.sum(dim=1)
    cond = (tot > 1 - atol) * (tot < 1 + atol)
    cond = cond.prod()
    assert int(cond.data.cpu().numpy()) == 1, 'Probability normalization check failed.'
