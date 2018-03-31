# -*- coding: utf-8 -*-
# File   : parameter.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 31/03/2018
# 
# This file is part of Jacinle.

import six

from jacinle.utils.matching import NameMatcher

__all__ = ['find_parameters', 'filter_parameters', 'exclude_parameters']


def find_parameters(module, pattern, return_names=False):
    return filter_parameters(module.named_parameters(), pattern, return_names=return_names)


def filter_parameters(params, pattern, return_names=False):
    if isinstance(pattern, six.string_types):
        pattern = [pattern]
    matcher = NameMatcher({p: True for p in pattern})
    with matcher:
        if return_names:
            return [(name, p) for name, p in params if matcher.match(name)]
        else:
            return [p for name, p in params if matcher.match(name)]


def exclude_parameters(params, exclude):
    return [p for p in params if p not in exclude]
