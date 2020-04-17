#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : layer.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 04/20/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from jacinle.logging import get_logger
from jactorch.functional import length2mask, mask_meshgrid

from .modules.dimension import Expander, Reducer, Permutation
from .modules.neural_logic import NeuralLogicInference, NeuralLogicInferenceMethod

__all__ = ['NeuralLogicLayer', 'NeuralLogicMachine']

logger = get_logger(__file__)


def _get_tuple_n(x, n, tp):
    assert tp is not list
    if type(x) is tp:
        x = [x, ] * n
    assert len(x) == n, 'parameters should be {} or list of N elements'.format(tp)
    for i in x:
        assert type(i) is tp, 'elements of list should be {}'.format(tp)
    return x


class NeuralLogicLayer(nn.Module):
    def __init__(
            self, breadth, input_dims, output_dims,
            logic_model, logic_hidden_dim,
            exclude_self=True, residual=False,
            activation='sigmoid',  # neural logic model
            use_exists=True, min_val=0., max_val=1., # neural reduction
    ):
        super().__init__()
        if breadth > 3:
            logger.warn('Using LogicLayer with breadth > 3 may cause speed and memory issue.')

        self.max_order = breadth
        self.residual = residual

        input_dims = _get_tuple_n(input_dims, self.max_order + 1, int)
        output_dims = _get_tuple_n(output_dims, self.max_order + 1, int)

        self.logic, self.dim_perms, self.dim_expanders, self.dim_reducers = [nn.ModuleList() for _ in range(4)]
        for i in range(self.max_order + 1):
            current_dim = input_dims[i]
            if i > 0:
                expander = Expander(i - 1)
                self.dim_expanders.append(expander)
                current_dim += expander.get_output_dim(input_dims[i - 1])
            else:
                self.dim_expanders.append(None)

            if i + 1 < self.max_order + 1:
                reducer = Reducer(i + 1, exclude_self, exists=use_exists, min_val=min_val, max_val=max_val)
                self.dim_reducers.append(reducer)
                current_dim += reducer.get_output_dim(input_dims[i + 1])
            else:
                self.dim_reducers.append(None)

            if current_dim == 0 or output_dims[i] == 0:
                self.dim_perms.append(None)
                self.logic.append(None)
                output_dims[i] = 0
            else:
                perm = Permutation(i)
                self.dim_perms.append(perm)
                current_dim = perm.get_output_dim(current_dim)
                self.logic.append(NeuralLogicInference(logic_model, current_dim, output_dims[i], logic_hidden_dim, activation))

        self.input_dims = input_dims
        self.output_dims = output_dims

        if self.residual:
            for i in range(len(input_dims)):
                self.output_dims[i] += input_dims[i]

    def forward(self, inputs, inputs_length_or_mask=None):
        assert len(inputs) > 1, 'Does not support breadth == 0.'
        assert len(inputs) == self.max_order + 1
        outputs = []

        inputs_length_mask = None
        if inputs_length_or_mask is not None:
            if inputs_length_or_mask.dim() == 1:
                inputs_length_mask = length2mask(inputs_length, inputs.size(1))
            else:
                inputs_length_mask = inputs_length_or_mask

        for i in range(self.max_order + 1):
            f = []
            if i > 0 and self.input_dims[i - 1] > 0:
                n = inputs[i].size(1) if i == 1 else None
                f.append(self.dim_expanders[i](inputs[i - 1], n))
            if i < len(inputs) and self.input_dims[i] > 0:
                f.append(inputs[i])
            if i + 1 < len(inputs) and self.input_dims[i + 1] > 0:
                mask = None
                if inputs_length_mask is not None:
                    mask = mask_meshgrid(inputs_length_mask, i + 1)
                f.append(self.dim_reducers[i](inputs[i + 1], mask))

            if len(f) == 0 or self.output_dims[i] == 0:
                output = None
            else:
                f = torch.cat(f, dim=-1)
                f = self.dim_perms[i](f)
                output = self.logic[i](f)
            if self.residual and self.input_dims[i] > 0:
                output = torch.cat([inputs[i], output], dim=-1)
            outputs.append(output)
        return outputs

    __hyperparams__ = (
        'breadth', 'input_dims', 'logic_model', 'output_dims', 'logic_hidden_dim',
        'exclude_self', 'residual', 'activation'
    )

    __hyperparam_defaults__ = {
        'logic_model': 'mlp',
        'exclude_self': True,
        'residual': False,
        'activation': 'sigmoid'
    }

    @classmethod
    def make_prog_block_parser(cls, parser, defaults, prefix=None):
        for k, v in cls.__hyperparam_defaults__.items():
            defaults.setdefault(k, v)

        if prefix is None:
            prefix = '--'
        else:
            prefix = '--' + str(prefix) + '-'

        parser.add_argument(prefix + 'breadth', type='int',
                            default=defaults['breadth'], metavar='N')
        parser.add_argument(prefix + 'logic-model',
                            default=defaults['logic_model'], choices=NeuralLogicInferenceMethod.choice_values(), metavar='T')
        parser.add_argument(prefix + 'logic-hidden-dim', type=int, nargs='+',
                            default=defaults['logic_hidden_dim'], metavar='N')
        parser.add_argument(prefix + 'exclude-self', type='bool',
                            default=defaults['exclude_self'], metavar='B')
        parser.add_argument(prefix + 'residual', type='bool',
                            default=defaults['residual'], metavar='B')
        parser.add_argument(prefix + 'activation', choices=['sigmoid', 'tanh', 'relu'],
                            default=defaults['activation'])

    @classmethod
    def from_args(cls, input_dims, output_dims, args, prefix=None, **kwargs):
        if prefix is None:
            prefix = ''
        else:
            prefix = str(prefix) + '_'

        setattr(args, prefix + 'input_dims', input_dims)
        setattr(args, prefix + 'output_dims', output_dims)
        init_params = {k: getattr(args, prefix + k) for k in cls.__hyperparams__}
        init_params.update(kwargs)

        return cls(**init_params)


class NeuralLogicMachine(nn.Module):
    def __init__(
            self, depth, breadth, input_dims, output_dims,
            logic_model, logic_hidden_dim, exclude_self=True,
            residual=False, io_residual=False, connections=None, activation='sigmoid',
            min_val=0., max_val=1., use_exists=True
    ):
        super().__init__()
        self.depth = depth
        self.breadth = breadth
        self.residual = residual
        self.io_residual = io_residual
        self.connections = connections

        self.input_dims = input_dims
        self.output_dims = output_dims

        assert not (self.residual and self.io_residual)

        def add_(x, y):
            for i in range(len(y)):
                x[i] += y[i]
            return x

        self.layers = nn.ModuleList()
        current_dims = input_dims
        total_output_dims = [0 for _ in range(self.breadth + 1)]  # for IO residual only
        for i in range(depth):
            if i > 0 and io_residual:
                add_(current_dims, input_dims)
            layer = NeuralLogicLayer(breadth, current_dims, output_dims, logic_model, logic_hidden_dim, exclude_self, residual,
                                     activation=activation, min_val=min_val, max_val=max_val, use_exists=use_exists)
            current_dims = layer.output_dims
            current_dims = self._mask(current_dims, i, 0)
            if io_residual:
                add_(total_output_dims, current_dims)
            self.layers.append(layer)

        if io_residual:
            self.output_dims = total_output_dims
        else:
            self.output_dims = current_dims

    def _mask(self, a, i, masked_value):
        if self.connections is not None:
            assert i < len(self.connections)
            mask = self.connections[i]
            if mask is not None:
                assert len(mask) == len(a)
                a = [x if y else masked_value for x, y in zip(a, mask)]
        return a

    def forward(self, inputs, inputs_length_or_mask=None, depth=None):
        outputs = [None for _ in range(self.breadth + 1)]
        f = inputs

        if depth is None:
            depth = self.depth
            assert depth <= self.depth

        def merge(x, y):
            if x is None:
                return y
            if y is None:
                return x
            return torch.cat([x, y], dim=-1)

        layer = None
        for i in range(depth):
            if i > 0 and self.io_residual:
                for j, inp in enumerate(inputs):
                    f[j] = merge(f[j], inp)
            layer = self.layers[i]
            f = layer(f, inputs_length_or_mask)
            f = self._mask(f, i, None)
            if self.io_residual:
                for j, out in enumerate(f):
                    outputs[j] = merge(outputs[j], out)
        if not self.io_residual:
            outputs = f
        return outputs

    __hyperparams__ = (
        'depth', 'breadth', 'input_dims', 'output_dims', 'logic_model', 'logic_hidden_dim',
        'exclude_self', 'io_residual', 'residual', 'activation'
    )

    __hyperparam_defaults__ = {
        'logic_model': 'mlp',
        'exclude_self': True,
        'io_residual': False,
        'residual': False,
        'activation': 'sigmoid'
    }

    @classmethod
    def make_prog_block_parser(cls, parser, defaults, prefix=None):
        for k, v in cls.__hyperparam_defaults__.items():
            defaults.setdefault(k, v)

        if prefix is None:
            prefix = '--'
        else:
            prefix = '--' + str(prefix) + '-'

        parser.add_argument(prefix + 'depth', type=int,
                            default=defaults['depth'], metavar='N')
        parser.add_argument(prefix + 'breadth', type=int,
                            default=defaults['breadth'], metavar='N')
        parser.add_argument(prefix + 'logic-model',
                            default=defaults['logic_model'], choices=NeuralLogicInferenceMethod.choice_values(), metavar='T')
        parser.add_argument(prefix + 'logic-hidden-dim', type=int, nargs='+',
                            default=defaults['logic_hidden_dim'], metavar='N')
        parser.add_argument(prefix + 'exclude-self', type='bool',
                            default=defaults['exclude_self'], metavar='B')
        parser.add_argument(prefix + 'io_residual', type='bool',
                            default=defaults['io_residual'], metavar='B')
        parser.add_argument(prefix + 'residual', type='bool',
                            default=defaults['residual'], metavar='B')
        parser.add_argument(prefix + 'activation', choices=['sigmoid', 'tanh', 'relu'],
                            default=defaults['activation'])

    @classmethod
    def from_args(cls, input_dims, output_dims, args, prefix=None, **kwargs):
        if prefix is None:
            prefix = ''
        else:
            prefix = str(prefix) + '_'

        setattr(args, prefix + 'input_dims', input_dims)
        setattr(args, prefix + 'output_dims', output_dims)
        init_params = {k: getattr(args, prefix + k) for k in cls.__hyperparams__}
        init_params.update(kwargs)

        return cls(**init_params)
