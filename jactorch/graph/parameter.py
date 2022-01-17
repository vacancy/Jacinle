#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : parameter.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/31/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import six
import contextlib

import torch.nn as nn
from jacinle.logging import get_logger
from jacinle.utils.matching import NameMatcher

logger = get_logger(__file__)

__all__ = [
    'find_parameters', 'filter_parameters', 'exclude_parameters', 'compose_param_groups', 'param_group',
    'mark_freezed', 'mark_unfreezed', 'detach_modules'
]


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


def compose_param_groups(model, *groups, filter_grad=True, verbose=True):
    """
    Compose the param_groups argument for torch optimizers.

    Examples:

    >>> optim.Adam(compose_param_groups(
    >>>     param_group('*.weight', lr=0.01)
    >>> ), lr=0.1)

    Args:
        model: the model containing optimizable variables.
        *groups: groups defined by patterns, of form (pattern, special_params)
        filter_grad: only choose parameters with requires_grad=True

    Returns:
        param_groups argument that can be passed to torch optimizers.

    """
    matcher = NameMatcher([(g[0], i) for i, g in enumerate(groups)])
    param_groups = [{'params': [], 'names': []} for _ in range(len(groups) + 1)]
    with matcher:
        for name, p in model.named_parameters():
            if filter_grad and not p.requires_grad:
                continue

            res = matcher.match(name)
            if res is None:
                res = -1
            param_groups[res]['names'].append(name)
            param_groups[res]['params'].append(p)
    for i, g in enumerate(groups):
        param_groups[i].update(g[1])

    if verbose:
        print_info = ['Param groups:']
        for group in param_groups:
            extra_params = ['{}: {}'.format(key, value) for key, value in group.items() if key not in ('params', 'names')]
            extra_params = '; '.join(extra_params)
            if extra_params == '':
                extra_params = '(default)'
            for name in group['names']:
                print_info.append('  {name}: {extra}.'.format(name=name, extra=extra_params))
        logger.info('\n'.join(print_info))

    return param_groups


def param_group(pattern, **kwargs):
    """A helper function used for human-friendly declaration of param groups."""
    return (pattern, kwargs)


def mark_freezed(model):
    model.eval()  # Turn off all BatchNorm / Dropout
    for p in model.parameters():
        p.requires_grad = False


def mark_unfreezed(model):
    model.train()  # Turn on all BatchNorm / Dropout
    for p in model.parameters():
        p.requires_grad = True


@contextlib.contextmanager
def detach_modules(*modules):
    all_modules = nn.ModuleList(modules)
    current_values = dict()
    for name, p in all_modules.named_parameters():
        current_values[name] = p.requires_grad
        p.requires_grad = False
    yield
    for name, p in all_modules.named_parameters():
        p.requires_grad = current_values[name]

