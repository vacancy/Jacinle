#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : io.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os.path as osp

import numpy as np
import torch
import torch.nn as nn

import jacinle.io as io
from jacinle.logging import get_logger
from jacinle.utils.matching import IENameMatcher
from jactorch.utils.meta import as_cpu

logger = get_logger(__file__)

__all__ = ['load_state_dict', 'load_weights']


__extra_magic_name__ = '__jacinle_extra_state_dict__'


def state_dict(model, include=None, exclude=None, cpu=True):
    if isinstance(model, nn.DataParallel):
        model = model.module

    state_dict = model.state_dict()

    matcher = IENameMatcher(include, exclude)
    with matcher:
        state_dict = {k: v for k, v in state_dict.items() if matcher.match(k)}
    stat = matcher.get_last_stat()
    if len(stat[1]) > 0:
        logger.critical('Weights {}: {}.'.format(stat[0], ', '.join(sorted(list(stat[1])))))

    if cpu:
        state_dict = as_cpu(state_dict)

    if hasattr(model, 'extra_state_dict'):
        state_dict[__extra_magic_name__] = model.extra_state_dict()

    return state_dict


def load_state_dict(model, state_dict, include=None, exclude=None):
    if isinstance(model, nn.DataParallel):
        model = model.module

    extra_state_dict = None
    if __extra_magic_name__ in state_dict:
        extra_state_dict = state_dict.pop(__extra_magic_name__)

    matcher = IENameMatcher(include, exclude)
    with matcher:
        state_dict = {k: v for k, v in state_dict.items() if matcher.match(k)}
    stat = matcher.get_last_stat()
    if len(stat[1]) > 0:
        logger.critical('Weights {}: {}.'.format(stat[0], ', '.join(sorted(list(stat[1])))))

    # Build the tensors.
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            state_dict[k] = torch.from_numpy(v)

    error_msg = []
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                error_msg.append('While copying the parameter named {}, '
                                 'whose dimensions in the model are {} and '
                                 'whose dimensions in the checkpoint are {}.'
                                 .format(name, own_state[name].size(), param.size()))

    if extra_state_dict is not None:
        if hasattr(model, 'load_extra_state_dict'):
            model.load_extra_state_dict(extra_state_dict)
        else:
            logger.warning('Extra state dict found but the model does not support load_extra_state_dict.')

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        error_msg.append('Missing keys in state_dict: "{}".'.format(missing))

    unexpected = set(state_dict.keys()) - set(own_state.keys())
    if len(unexpected) > 0:
        error_msg.append('Unexpected key "{}" in state_dict.'.format(unexpected))

    if len(error_msg):
        raise KeyError('\n'.join(error_msg))


def load_weights(model, filename, include=None, exclude=None):
    if osp.isfile(filename):
        try:
            weights = io.load(filename)
            # Hack for checkpoint.
            if 'model' in weights and 'optimizer' in weights:
                weights = weights['model']

            try:
                load_state_dict(model, weights, include=include, exclude=exclude)
            except KeyError as e:
                logger.warning('Unexpected or missing weights found:\n' + e.args[0])
            logger.critical('Weights loaded: {}.'.format(filename))
            return True
        except Exception:
            logger.exception('Error occurred when load weights {}.'.format(filename))
    else:
        logger.warning('No weights file found at specified position: {}.'.format(filename))
    return None

