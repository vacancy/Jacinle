#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : collate.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import re
import collections

import torch
import torch.utils.data.dataloader as torchdl

from six import string_types

from jacinle.utils.argument import UniqueValueGetter
from jacinle.utils.enum import JacEnum

__all__ = ['user_scattered_collate', 'VarLengthCollateMode', 'VarLengthCollate', 'VarLengthCollateV2']

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
    return batch


class VarLengthCollateMode(JacEnum):
    PAD = 'pad'
    CONCAT = 'concat'
    PAD2D = 'pad2d'


class VarLengthCollate(object):
    def __init__(self, fields, mode='pad'):
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
        if torchdl._use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            if process and self._mode is VarLengthCollateMode.PAD:
                numel = max([x.numel() for x in values]) * len(values)
            else:
                numel = sum([x.numel() for x in values])
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


class VarLengthCollateV2(object):
    def __init__(self, fields, mode='pad'):
        self._fields = fields
        self._mode = VarLengthCollateMode.from_string(mode)

    def __call__(self, batch, key=None):
        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        elem_type = type(batch[0])

        if key is not None:
            assert torch.is_tensor(batch[0]) or (elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_'
                                                 and elem_type.__name__ != 'string_')

        if torch.is_tensor(batch[0]):
            return self._stack(batch, key)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))

                return self._stack([torch.from_numpy(b) for b in batch], key)
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
                    values, lengths = self(values, key=key)
                    result[key] = values
                    result[key + '_length'] = lengths
                else:
                    result[key] = self(values)
            return result
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [self(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    def _stack(self, values, key=None):
        out = None
        if torchdl._use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            if key is not None:
                if self._mode is VarLengthCollateMode.PAD:
                    numel = max([x.numel() for x in values]) * len(values)
                elif self._mode is VarLengthCollateMode.CONCAT:
                    numel = sum([x.numel() for x in values])
            else:
                numel = sum([x.numel() for x in values])
            storage = values[0].storage()._new_shared(numel)
            out = values[0].new(storage)

        if key is None:
            return torch.stack(values, 0, out=out)

        mode_spec = self._fields[key]
        if isinstance(mode_spec, tuple):
            mode = VarLengthCollateMode.from_string(mode_spec[0])
            parameters = mode_spec[1:]
        else:
            mode = VarLengthCollateMode.from_string(mode_spec)
            parameters = tuple()

        if mode is VarLengthCollateMode.CONCAT:
            uvg = UniqueValueGetter('Tensor sizes should match except the first dim.')
            for v in values:
                uvg.set(v.size()[1:])
            lengths = [v.size(0) for v in values]
            return torch.cat(values, 0, out=out), torch.LongTensor(lengths)
        elif mode is VarLengthCollateMode.PAD:
            uvg = UniqueValueGetter('Tensor sizes should match except the first dim.')
            for v in values:
                uvg.set(v.size()[1:])
            pad_value = parameters[0] if len(parameters) > 0 else 0

            lengths = [v.size(0) for v in values]
            max_length = max(lengths)
            result = []
            for v in values:
                if v.size(0) < max_length:
                    v = torch.cat([v, v.new(*((max_length - v.size(0), ) + v.size()[1:])).fill_(pad_value)], dim=0)
                result.append(v)
            return torch.stack(result, 0, out=out), torch.LongTensor(lengths)
        elif mode is VarLengthCollateMode.PAD2D:
            uvg = UniqueValueGetter('Tensor sizes should match except the first 2 dims.')
            for v in values:
                uvg.set(v.size()[2:])
            pad_value = parameters[0] if len(parameters) > 0 else 0

            lengths = [v.size()[:2] for v in values]
            max_h, max_w = max([x[0] for x in lengths]), max([x[1] for x in lengths])
            result = []
            for v in values:
                u = v.new(max_h, max_w, *uvg.get()).fill_(pad_value)
                u[:] = pad_value
                u[:v.size(0), :v.size(1)] = v
                result.append(u)
            return torch.stack(result, 0, out=out), torch.LongTensor(lengths)
        else:
            raise ValueError('Unknown collation mode: {}.'.format(self._mode))
