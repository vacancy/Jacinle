#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : collate_v2.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/09/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import re
import collections

import torch

from six import string_types

from jacinle.utils.argument import UniqueValueGetter
from jacinle.utils.enum import JacEnum
from .utils import use_shared_memory, numpy_type_map, user_scattered_collate, VarLengthCollateMode

__all__ = ['VarLengthCollateV2']


class VarLengthCollateV2(object):
    """
    Collate a batch of data from multiple workers.
    It supports data of variant length. For example, a batch may contain sentences of different length to be
    processed using LSTM models. Usually, we choose the pad the shorter sentences to make them of the same length.
    Thus, they can be processed in a batch.

    To archive this, this module provides a fine-grained collate control over each input field and supports multiple
    ways for collating the data. It assumes that the input data is a dict. Example:

    >>> collate_fn = VarLengthCollateV2({'sentence': 'pad', 'image': 'padimage'})
    >>> collate_fn({
    >>>     'sentence': [torch.rand(3), torch.rand(4)],
    >>>     'image': [torch.rand(3, 16, 14), torch.rand(3, 8, 12)]
    >>> })

    It can be directly passed to the DataLaoder as the parameter `collate_fn`.

    >>> from torch.utils.data.dataloader import DataLoader
    >>> from torch.utils.data.dataset import Dataset
    >>> dataset = Dataset()
    >>> collate_fn = VarLengthCollateV2({'sentence': 'pad', 'image': 'padimage'})
    >>> dataloader = DataLoader(dataset, collate_fn=collate_fn)

    Here is a complete list of the supported collate mode:

    1. skip: the field will be skipped, no collation will be done. This is useful when sometimes you are trasmitting
    some meta information to the model.
    2. concat: assume the data is one-dimentional. The data will be concatenated along this dimension.
    3. pad: assume the data is one-dimensional. The data will be padded into the same length (the maximum length of all
    data) and get concatenated along a new dimension.
    4. pad2d: similar to the pad mode, it takes 2d inputs (h, w) and pads them.
    5. padimage: similar to the pad2d, except that it takes 3d inputs (d, h, w), where the d dimension will not be
    padded.

    """
    def __init__(self, fields):
        self._fields = fields

    def __call__(self, batch, key=None):
        error_msg = "Batch must contain tensors, numbers, dicts or lists; found {}."
        elem_type = type(batch[0])

        if key is not None:
            assert torch.is_tensor(batch[0]) or (elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_'
                                                 and elem_type.__name__ != 'string_'), 'Invalid field: {}.'.format(key)

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
                    if isinstance(self._fields[key], string_types) and VarLengthCollateMode.from_string(self._fields[key]) is VarLengthCollateMode.SKIP:
                        result[key] = values
                    else:
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
        mode, parameters = None, None
        if key is not None:
            mode_spec = self._fields[key]
            if isinstance(mode_spec, tuple):
                mode = VarLengthCollateMode.from_string(mode_spec[0])
                parameters = mode_spec[1:]
            else:
                mode = VarLengthCollateMode.from_string(mode_spec)
                parameters = tuple()

        out = None
        if use_shared_memory():
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = 0
            if key is not None:
                if mode is VarLengthCollateMode.PAD:
                    numel = max([x.numel() for x in values]) * len(values)
                elif mode is VarLengthCollateMode.CONCAT:
                    numel = sum([x.numel() for x in values])
                elif mode is VarLengthCollateMode.PAD2D:
                    max_h = max([x.size(0) for x in values])
                    max_w = max([x.size(1) for x in values])
                    hw = max_h * max_w
                    numel = sum([x.numel() // x.size(0) // x.size(1) * hw for x in values])
            else:
                numel = sum([x.numel() for x in values])

            if numel > 0:
                storage = values[0].storage()._new_shared(numel)
                out = values[0].new(storage)

        if key is None:
            return torch.stack(values, 0, out=out)

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
            rest_size = uvg.get() or []
            pad_value = parameters[0] if len(parameters) > 0 else 0

            lengths = [v.size()[:2] for v in values]
            max_h, max_w = max([x[0] for x in lengths]), max([x[1] for x in lengths])
            result = []
            for v in values:
                u = v.new(*(max_h, max_w, *rest_size)).fill_(pad_value)
                u[:v.size(0), :v.size(1)] = v
                result.append(u)
            return torch.stack(result, 0, out=out), torch.LongTensor(lengths)
        elif mode is VarLengthCollateMode.PADIMAGE:
            uvg = UniqueValueGetter('Tensor sizes should match except the last 2 dims.')
            for v in values:
                assert v.dim() == 3, 'Support only 3-dimensional input.'
                uvg.set(v.size(0))
            pad_value = parameters[0] if len(parameters) > 0 else 0

            lengths = [v.size()[-2:] for v in values]
            max_h, max_w = max([x[0] for x in lengths]), max([x[1] for x in lengths])
            result = []
            for v in values:
                u = v.new(*(uvg.get(), max_h, max_w)).fill_(pad_value)
                # TODO(Jiayuan Mao @ 07/19): support input with dim > 3.
                u[:, :v.size(1), :v.size(2)] = v
                result.append(u)
            return torch.stack(result, 0, out=out), torch.LongTensor(lengths)
        else:
            raise ValueError('Unknown collation mode: {}.'.format(mode))

