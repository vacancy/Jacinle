# -*- coding: utf-8 -*-
# File   : copy.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 24/01/2018
# 
# This file is part of Jacinle.

import collections

import torch

__all__ = ['async_copy_to']


def async_copy_to(obj, dev, main_stream=None):
    # Adapted from: https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/_functions.py
    if torch.is_tensor(obj):
        v = obj.cuda(dev, non_blocking=True)
        if main_stream is not None:
            v.record_stream(main_stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, dev, main_stream) for k, o in obj.items()}
    elif isinstance(obj, (tuple, list, collections.UserList)):
        return [async_copy_to(o, dev, main_stream) for o in obj]
    else:
        return obj
