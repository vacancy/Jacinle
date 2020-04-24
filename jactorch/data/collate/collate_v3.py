#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : collate_v3.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com # Date   : 03/04/2018
# Date   : 04/04/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import re
import collections

import torch

from six import string_types

from torch.nn.parallel._functions import Gather
from jacinle.utils.argument import UniqueValueGetter
from jacinle.utils.enum import JacEnum
from jactorch.data.layout import DataLayout, DataLayoutType
from .utils import use_shared_memory, numpy_type_map

__all__ = [
    'VarLengthCollateV3'
]


class _VarLengthCollateV3Stack(object):
    def apply(self, feed_dict, key):
        raise NotImplementedError()


class _VarLengthCollateV3ArrayStack(_VarLengthCollateV3Stack):
    def __init__(self, array, length):
        self.array = array
        self.length = length

    def apply(self, feed_dict, key):
        feed_dict[key] = self.array
        feed_dict[key + '_length'] = self.length


class VarLengthCollateV3Mode(JacEnum):
    COLLATE = 'collate'  # for data loader collating.
    GATHER = 'gather'   # for data parallel gathering.


class VarLengthCollateV3(object):
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
    def __init__(
        self, layout, mode='collate',
        gather_device=None, gather_dim=0,
    ):

        self.layout = layout
        if isinstance(self.layout, dict):
            self.layout = DataLayout(self.layout)
        self.mode = VarLengthCollateV3Mode.from_string(mode)
        self.gather_device = gather_device
        self.gather_dim = gather_dim

    def __call__(self, batch, flatten_key=None, layout_spec=None):
        if flatten_key is not None and flatten_key in self.layout:
            layout_spec = self.layout[flatten_key]

        if layout_spec is not None and layout_spec.type is DataLayoutType.SKIP:
            return batch

        error_msg = "Batch must contain tensors, numbers, dicts or lists; found {}."
        elem_type = type(batch[0])
        if layout_spec is not None:
            assert (
                torch.is_tensor(batch[0]) or
                (elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_')
            ), 'Invalid layout type for: {}.'.format(flatten_key)

        if torch.is_tensor(batch[0]):
            return self._stack(batch, layout_spec, maybe_cuda=True)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))
                return self._stack([torch.from_numpy(b) for b in batch], layout_spec, maybe_cuda=False)
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
            result = dict()
            for key in batch[0]:
                values = [d[key] for d in batch]
                next_key = key if flatten_key is None else f'{flatten_key}.{key}'
                values = self(values, flatten_key=next_key, layout_spec=layout_spec)
                if isinstance(values, _VarLengthCollateV3Stack):
                    values.apply(result, key)
                else:
                    result[key] = values
            return result
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            # Add .{index} only if it's inside a dict already.
            return [
                self(samples, flatten_key=None if flatten_key is None else f'{flatten_key}.{i}',
                     layout_spec=layout_spec)
                for i, samples in enumerate(transposed)
            ]

        raise TypeError((error_msg.format(type(batch[0]))))

    def _stack_raw(self, values, out, maybe_cuda, is_concat=False):
        if self.mode is VarLengthCollateV3Mode.GATHER and maybe_cuda:
            if values[0].dim() == 0:
                values = [o.unsqueeze(0) for o in values]
            return Gather.apply(self.gather_device, self.gather_dim, *values)
        else:
            if is_concat:
                return torch.cat(values, 0, out=out)
            else:
                return torch.stack(values, 0, out=out)

    def _stack(self, values, spec=None, maybe_cuda=True):
        mode = spec.type if spec is not None else None

        out = None
        if self.mode is VarLengthCollateV3Mode.COLLATE and use_shared_memory():
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = 0
            if mode is not None:
                if mode is DataLayoutType.CONCAT:
                    numel = sum([x.numel() for x in values])
                elif mode is DataLayoutType.PAD:
                    numel = max([x.numel() for x in values]) * len(values)
                elif mode is DataLayoutType.PAD2D:
                    max_h = max([x.size(0) for x in values])
                    max_w = max([x.size(1) for x in values])
                    hw = max_h * max_w
                    numel = sum([x.numel() // x.size(0) // x.size(1) * hw for x in values])
                elif mode is DataLayoutType.PADIMAGE:
                    max_h = max([x.size(1) for x in values])
                    max_w = max([x.size(2) for x in values])
                    hw = max_h * max_w
                    numel = sum([x.numel() // x.size(1) // x.size(2) * hw for x in values])
            else:
                numel = sum([x.numel() for x in values])

            if numel > 0:
                storage = values[0].storage()._new_shared(numel)
                out = values[0].new(storage)

        if mode is None:
            return self._stack_raw(values, out=out, maybe_cuda=maybe_cuda)

        if mode is DataLayoutType.CONCAT:
            uvg = UniqueValueGetter('Tensor sizes should match except the first dim.')
            for v in values:
                uvg.set(v.size()[1:])
            lengths = [v.size(0) for v in values]
            return _VarLengthCollateV3ArrayStack(self._stack_raw(values, out=out, maybe_cuda=maybe_cuda, is_concat=True), torch.LongTensor(lengths))
        elif mode is DataLayoutType.PAD:
            uvg = UniqueValueGetter('Tensor sizes should match except the first dim.')
            for v in values:
                uvg.set(v.size()[1:])
            pad_value = spec.fill

            lengths = [v.size(0) for v in values]
            max_length = max(lengths)
            result = []
            for v in values:
                if v.size(0) < max_length:
                    v = torch.cat([v, v.new(*((max_length - v.size(0), ) + v.size()[1:])).fill_(pad_value)], dim=0)
                result.append(v)
            return _VarLengthCollateV3ArrayStack(self._stack_raw(result, out=out, maybe_cuda=maybe_cuda), torch.LongTensor(lengths))
        elif mode is DataLayoutType.PAD2D:
            uvg = UniqueValueGetter('Tensor sizes should match except the first 2 dims.')
            for v in values:
                uvg.set(v.size()[2:])
            rest_size = uvg.get() or []
            pad_value = spec.fill

            lengths = [v.size()[:2] for v in values]
            max_h, max_w = max([x[0] for x in lengths]), max([x[1] for x in lengths])
            result = []
            for v in values:
                u = v.new(*(max_h, max_w, *rest_size)).fill_(pad_value)
                u[:v.size(0), :v.size(1)] = v
                result.append(u)
            return _VarLengthCollateV3ArrayStack(self._stack_raw(result, out=out, maybe_cuda=maybe_cuda), torch.LongTensor(lengths))
        elif mode is DataLayoutType.PADIMAGE:
            uvg = UniqueValueGetter('Tensor sizes should match except the last 2 dims.')
            for v in values:
                assert v.dim() == 3, 'Support only 3-dimensional input.'
                uvg.set(v.size(0))
            pad_value = spec.fill

            lengths = [v.size()[-2:] for v in values]
            max_h, max_w = max([x[0] for x in lengths]), max([x[1] for x in lengths])
            result = []
            for v in values:
                u = v.new(*(uvg.get(), max_h, max_w)).fill_(pad_value)
                # TODO(Jiayuan Mao @ 07/19): support input with dim > 3.
                u[:, :v.size(1), :v.size(2)] = v
                result.append(u)
            return _VarLengthCollateV3ArrayStack(self._stack_raw(result, out=out, maybe_cuda=maybe_cuda), torch.LongTensor(lengths))
        else:
            raise ValueError('Unknown collation mode: {}.'.format(mode))

