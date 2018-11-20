#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : probability.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['normalize_prob', 'check_prob_normalization']


def normalize_prob(a, dim=-1):
    """Perform 1-norm along the specific dimension."""
    return a / a.sum(dim=dim, keepdim=True)


def check_prob_normalization(p, dim=-1, atol=1e-5):
    """Check if the probability is normalized along a specific dimension."""
    tot = p.sum(dim=dim)
    cond = (tot > 1 - atol) * (tot < 1 + atol)
    cond = cond.prod()
    assert int(cond.data.cpu().numpy()) == 1, 'Probability normalization check failed.'
