#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : batch.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections
import numpy as np

__all__ = ['batchify', 'unbatchify']


def batchify(inputs):
    first = inputs[0]
    if isinstance(first, (tuple, list, collections.UserList)):
        return [batchify([ele[i] for ele in inputs]) for i in range(len(first))]
    elif isinstance(first, (collections.Mapping, collections.UserDict)):
        return {k: batchify([ele[k] for ele in inputs]) for k in first}
    return np.stack(inputs)


def unbatchify(inputs):
    if isinstance(inputs, (tuple, list, collections.UserList)):
        outputs = [unbatchify(e) for e in inputs]
        return list(map(list, zip(*outputs)))
    elif isinstance(inputs, (collections.Mapping, collections.UserDict)):
        outputs = {k: unbatchify(v) for k, v in inputs.items()}
        first = outputs[0]
        return [{k: outputs[k][i] for k in inputs} for i in range(len(first))]
    return list(inputs)
