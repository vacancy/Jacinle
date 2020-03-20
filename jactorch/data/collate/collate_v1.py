#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : collate_v1.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/21/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
Function for collate data generated from multiple workers.
It supports collating data with variant length.
"""

import re
import collections

import torch

from six import string_types

from jacinle.utils.argument import UniqueValueGetter
from .utils import use_shared_memory, numpy_type_map, VarLengthCollateMode

__all__ = ['VarLengthCollateV1']


class VarLengthCollateV1(object):
    def __init__(self, fields, mode='pad'):
        """
        VarLengthCollate is deprecated in favor of VarLengthCollateV2.

        Args:
            fields: the fields to be collated with variant length.
            mode: the collate mode.

        """
        self._fields = fields
        self._mode = VarLengthCollateMode.from_string(mode)

    def __call__(self, batch, process=False):
        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        elem_type = type(batch[0])

        if process:
            assert torch.is_tensor(batch[0]) or (elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_'
                                                 and elem_type.__name__ != 'string_')

        if torch.is_tensor(batch[0]):
            return self._stack(batch, process)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))

                return self._stack([torch.from_numpy(b) for b in batch], process)
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
        elif isinstance(batch[0], int):
            return torch.LongTensor(batch)
        elif isinstance(batch[0], float):
            return torch.DoubleTensor(batch)
        elif isinstance(batch[0], string_types):
            return batch
        elif isinstance(batch[0], collections.Mapping):
            result = {}
            for key in batch[0]:
                values = [d[key] for d in batch]
                if key in self._fields:
                    values, lengths = self(values, True)
                    result[key] = values
                    result[key + '_length'] = lengths
                else:
                    result[key] = self(values)
            return result
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [self(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    def _stack(self, values, process):
        uvg = UniqueValueGetter('Tensor sizes should match except the first dim.')
        for v in values:
            uvg.set(v.size()[1:])
        uvg.get()

        out = None
        if use_shared_memory():
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = 0
            if process:
                if self._mode is VarLengthCollateMode.PAD:
                    numel = max([x.numel() for x in values]) * len(values)
                elif self._mode is VarLengthCollateMode.CONCAT:
                    numel = sum([x.numel() for x in values])
                elif self._mode is VarLengthCollateMode.PAD2D:
                    max_h = max([x.size(0) for x in values])
                    max_w = max([x.size(1) for x in values])
                    hw = max_h * max_w
                    numel = sum([x.numel() // x.size(0) // x.size(1) * hw for x in values])
            else:
                numel = sum([x.numel() for x in values])

            if numel > 0:
                storage = values[0].storage()._new_shared(numel)
                out = values[0].new(storage)

        if not process:
            return torch.stack(values, 0, out=out)

        lengths = [v.size(0) for v in values]
        if self._mode is VarLengthCollateMode.CONCAT:
            return torch.cat(values, 0, out=out), torch.LongTensor(lengths)
        elif self._mode is VarLengthCollateMode.PAD:
            max_length = max(lengths)
            result = []
            for v in values:
                if v.size(0) < max_length:
                    v = torch.cat([v, v.new(*((max_length - v.size(0), ) + v.size()[1:])).zero_()], dim=0)
                result.append(v)
            return torch.stack(result, 0, out=out), torch.LongTensor(lengths)
        else:
            raise ValueError('Unknown collation mode: {}.'.format(self._mode))

