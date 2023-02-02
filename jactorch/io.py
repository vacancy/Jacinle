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
from typing import Any, Optional, Union, Sequence, Dict

import numpy as np
import torch
import torch.nn as nn

import jacinle.io as io
from jacinle.logging import get_logger
from jacinle.utils.matching import IENameMatcher
from jactorch.utils.meta import as_cpu

logger = get_logger(__file__)

__all__ = ['state_dict', 'load_state_dict', 'load_weights']


__extra_magic_name__ = '__jacinle_extra_state_dict__'


def state_dict(model: nn.Module, include: Optional[Sequence[str]] = None, exclude: Optional[Sequence[str]] = None, cpu: bool = True) -> Dict[str, Any]:
    """Get a state dict representation of the model. This function is similar to ``model.state_dict()``, but it also
    supports additional features including:

    - ``include`` and ``exclude``: only include/exclude some parameters.
    - ``cpu``: move the parameters to CPU.
    - ``extra_state_dict``: each module can implement a custom ``extra_state_dict`` method which return additional state
        of the module (such as hyperparameters, random seeds, etc.).

    Args:
        model: the PyTorch model.
        include: a list of parameter names to be included in the state dict (supports globbing).
        exclude: a list of parameter names to be excluded from the state dict (supports globbing).
        cpu: whether to move the parameters to CPU.

    Returns:
        A state dict.
    """

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

    extra_state_dict = dict()
    for name, module in model.named_modules():
        if hasattr(module, 'extra_state_dict'):
            extra_state_dict[name] = module.extra_state_dict()
    if len(extra_state_dict) > 0:
        state_dict[__extra_magic_name__] = extra_state_dict
    return state_dict


def load_state_dict(model: nn.Module, state_dict: Dict[str, Any], include: Optional[Sequence[str]] = None, exclude: Optional[Sequence[str]] = None):
    """Load a state dict into the model. This function is similar to ``model.load_state_dict()``, but it also
    supports additional features including:

    - ``include`` and ``exclude``: only include/exclude some parameters.
    - ``extra_state_dict``: each module can implement a custom ``load_extra_state_dict`` method which load additional state
        of the module (such as hyperparameters, random seeds, etc.).

    Furthermore, this function will not raise exceptions when there are missing or unexpected parameters. This is similar to the
    latest PyTorch behavior specifed by ``strict=False``.

    Args:
        model: the PyTorch model.
        state_dict: the state dict to be loaded.
        include: a list of parameter names to be included in the state dict (supports globbing).
        exclude: a list of parameter names to be excluded from the state dict (supports globbing).
    """

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
        if hasattr(model, 'load_extra_state_dict') and '' not in extra_state_dict:
            deprecated_warning = 'DEPRECATED(Jiayuan Mao): legacy extra_state_dict has been deprecated and will be removed by 01/15/2020; please update the old checkpoints.'
            logger.warning(deprecated_warning)
            model.load_extra_state_dict(extra_state_dict)
        else:
            name2module = dict(model.named_modules())
            for name, extra in extra_state_dict.items():
                module = name2module[name]
                if hasattr(module, 'load_extra_state_dict'):
                    module.load_extra_state_dict(extra)
                else:
                    logger.warning('Extra state dict found but the model does not support load_extra_state_dict: {}.'.format(name))

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        error_msg.append('Missing keys in state_dict: "{}".'.format(missing))

    unexpected = set(state_dict.keys()) - set(own_state.keys())
    if len(unexpected) > 0:
        error_msg.append('Unexpected key "{}" in state_dict.'.format(unexpected))

    if len(error_msg):
        raise KeyError('\n'.join(error_msg))


def load_weights(model, filename, include=None, exclude=None, return_raw=True) -> Union[bool, Optional[Dict[str, Any]]]:
    """Load weights from a file. Internally this function calls :func:`load_state_dict`.
    It handles the case where the loaded file is a checkpoint (i.e., containing both weights and optimizer states).
    It will automatically detect such case and extract the weights.

    Args:
        model: the PyTorch model.
        filename: the file name.
        include: a list of parameter names to be included in the state dict (supports globbing).
        exclude: a list of parameter names to be excluded from the state dict (supports globbing).
        return_raw: whether to return the raw state dict. If ``False``, this function will return ``True`` when the weights
            are successfully loaded. If ``True``, this function will return the loaded file (either a state dict or a checkpoint).
            when the weights are successfully loaded, and ``None`` otherwise.

    Returns:

        - If ``return_raw=False``, return ``True`` when the weights are successfully loaded, and ``False`` otherwise.
        - If ``return_raw=True``, return the loaded file (either a state dict or a checkpoint) when the weights are successfully loaded,
            and ``None`` otherwise.
    """
    if osp.isfile(filename):
        try:
            raw = weights = io.load(filename)
            # Hack for checkpoint.
            if 'model' in weights and 'optimizer' in weights:
                weights = weights['model']

            try:
                load_state_dict(model, weights, include=include, exclude=exclude)
            except KeyError as e:
                logger.warning('Unexpected or missing weights found:\n' + e.args[0])
            logger.critical('Weights loaded: {}.'.format(filename))
            if return_raw:
                return raw
            return True
        except Exception:
            logger.exception('Error occurred when load weights {}.'.format(filename))
    else:
        logger.warning('No weights file found at specified position: {}.'.format(filename))
    return None if return_raw else False

