# -*- coding: utf-8 -*-
# File   : copy.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 24/01/2018
# 
# This file is part of Jacinle.

import collections

import torch
from torch.autograd import Variable


def async_copy_to(obj, dev, main_stream=None):
    if isinstance(obj, Variable) or torch.is_tensor(obj):
        v = obj.cuda(dev, async=True)
        if main_stream is not None:
            v.data.record_stream(main_stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, dev, main_stream) for k, o in obj.items()}
    elif isinstance(obj, (tuple, list, collections.UserList)):
        return [async_copy_to(o, dev, main_stream) for o in obj]
    else:
        return obj
