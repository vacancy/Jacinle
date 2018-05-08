#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : stat.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/16/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['stat_histogram']


def stat_histogram(name, value, markers, value_format='{}'):
    histo = dict()
    for m in markers:
        histo[('{}/>=' + value_format).format(name, m)] = int(value >= m)
    return histo

