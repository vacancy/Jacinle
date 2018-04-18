# -*- coding: utf-8 -*-
# File   : stat.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/16/2018
# 
# This file is part of Jacinle.

__all__ = ['stat_histogram']


def stat_histogram(name, value, markers):
    histo = dict()
    for m in markers:
        histo['{}/>{}'.format(name, m)] = int(value > m)
    return histo

