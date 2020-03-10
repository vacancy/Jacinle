#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/09/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
from jacinle.utils.enum import JacEnum


def use_shared_memory():
    if torch.__version__ < '1.1':
        import torch.utils.data.dataloader as torchdl
        return torchdl._use_shared_memory
    elif torch.__version__ < '1.2':
        import torch.utils.data._utils.collate as torch_collate
        return torch_collate._use_shared_memory
    else:
        return torch.utils.data.get_worker_info() is not None


numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def user_scattered_collate(batch):
    """
    A helper function indicating that no collation needs to be done.
    """
    return batch


class VarLengthCollateMode(JacEnum):
    SKIP = 'skip'
    CONCAT = 'concat'
    PAD = 'pad'
    PAD2D = 'pad2d'
    PADIMAGE = 'padimage'

