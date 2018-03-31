# -*- coding: utf-8 -*-
# File   : io.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 24/01/2018
# 
# This file is part of Jacinle.

import os.path as osp

import numpy as np
import torch
import torch.nn as nn

import jacinle.io as io
from jacinle.logging import get_logger

logger = get_logger(__file__)

__all__ = ['load_state_dict', 'load_weights']


def load_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(), param.size()))

    error_msg = []

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        error_msg.append('Missing keys in state_dict: "{}".'.format(missing))

    unexpected = set(state_dict.keys()) - set(own_state.keys())
    if len(unexpected) > 0:
        error_msg.append('Unexpected key "{}" in state_dict.'.format(unexpected))

    if len(error_msg):
        raise KeyError('\n'.join(error_msg))


def load_weights(model, filename):
    if osp.isfile(filename):
        try:
            weights = io.load(filename)

            # Hack for checkpoint.
            if 'model' in weights and 'optimizer' in weights:
                weights = weights['model']

            # Build the tensors.
            for k, v in weights.items():
                if isinstance(v, np.ndarray):
                    weights[k] = torch.from_numpy(v)

            try:
                if isinstance(model, nn.DataParallel):
                    model = model.module
                load_state_dict(model, weights)
            except KeyError as e:
                logger.warning('Unexpected or missing weights found: {}.'.format(str(e)))
            logger.critical('Weights loaded: {}.'.format(filename))
            return True
        except Exception:
            logger.exception('Error occurred when load weights {}.'.format(filename))
    else:
        logger.warning('No weights file found at specified position: {}.'.format(filename))
    return None
