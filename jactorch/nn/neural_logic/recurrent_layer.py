#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : recurrent_layer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/10/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from jactorch.functional import concat_shape
from .modules.neural_logic import NeuralLogicInferenceMethod
from .modules.dimension import Reducer
from .layer import NeuralLogicMachine, _get_tuple_n

__all__ = ['RecurrentNeuralLogicMachine']


class RecurrentNeuralLogicMachine(nn.Module):
    def __init__(
            self, breadth,
            depth1, depth2, depth3,
            input_dims, imm_dims1, imm_dims2, output_dims,
            epsilon,
            logic_model, logic_hidden_dim,
            exclude_self=True, pre_residual=False, connections=None
    ):

        super().__init__()

        def add(x, y):
            """output[i] = x[i] + y[i]"""
            return [i + j for i, j in zip(x, y)]

        def mul(x, m):
            """output[i] = x[i] * m"""
            return [i * m for i in x]

        self.breadth = breadth
        self.input_dims = _get_tuple_n(input_dims, breadth + 1, int)
        self.imm_dims1 = _get_tuple_n(imm_dims1, breadth + 1, int)
        self.imm_dims2 = _get_tuple_n(imm_dims2, breadth + 1, int)
        self.output_dims = _get_tuple_n(output_dims, breadth + 1, int)

        kwargs = dict(logic_model=logic_model, logic_hidden_dim=logic_hidden_dim, exclude_self=exclude_self, connections=connections)
        if depth1 > 0:
            input_dims = self.input_dims
            output_dims = self.imm_dims1
            self.model_pre = NeuralLogicMachine(depth1, breadth, input_dims, output_dims, residual=pre_residual, **kwargs)
        else:
            self.add_module('model_pre', None)

        input_dims = add(self.model_pre.output_dims, mul(self.imm_dims2, 2))
        output_dims = mul(self.imm_dims2, 2)
        kwargs = dict(logic_model=logic_model, logic_hidden_dim=logic_hidden_dim, exclude_self=exclude_self, connections=connections)
        self.model = NeuralLogicMachine(depth2, breadth, input_dims, output_dims, residual=False, **kwargs)

        if depth3 > 0:
            input_dims = self.model.output_dims
            output_dims = self.output_dims
            kwargs = dict(logic_model=logic_model, logic_hidden_dim=logic_hidden_dim, exclude_self=exclude_self, connections=connections)
            self.model_post = NeuralLogicMachine(depth3, breadth, current_dims, self.output_dims, residual=False, **kwargs)
        else:
            self.add_module('model_post', None)

        self.reducers = nn.ModuleList()
        for i in range(self.breadth + 1):
            if i == 0:
                self.reducers.append(None)
            else:
                self.reducers.append(Reducer(i, exclude_self=exclude_self, exists=False))

        self.epsilon = epsilon

    def forward(self, inputs, max_depths=1000):
        for i in inputs:
            if i is not None:
                batch_size, n = i.size()[:2]
                dtype = i.dtype
                device = i.device
                break

        def zeros(d, h):
            """zeros(batch_size, ..., h)"""
            if d is None:
                return None
            return torch.zeros(concat_shape(batch_size, [n for _ in range(d)], h), dtype=dtype, device=device)

        def _format(x):
            for i in x:
                if i is None:
                    yield i
                else:
                    yield i.size()

        def print_list(*args, **kwargs):
            print(*[list(_format(x)) for x in args], **kwargs)

        def concat(x, y, z):
            def _concat():
                """output[i] = concat(x[i], y[i], z[i]). x[i] can be None"""
                for i, j, k in zip(x, y, z):
                    if i is None:
                        yield torch.cat((j, k), dim=-1)
                    else:
                        yield torch.cat((i, j, k), dim=-1)
            return list(_concat())

        def chunk(x, n, dim):
            def _chunk():
                for i in x:
                    yield i.chunk(n, dim=dim)
            chunks = list(_chunk())
            return list(zip(*chunks))

        def elem_op(x, y, op):
            def _add():
                """output[i] = x[i] + y[i]."""
                for i, j in zip(x, y):
                    yield op(i, j)
            return list(_add())

        import functools
        add = functools.partial(elem_op, op=lambda x, y: x + y)
        mul = functools.partial(elem_op, op=lambda x, y: x * y)
        div = functools.partial(elem_op, op=lambda x, y: x / y)

        def reduce(x, d):
            """return min(x). d is the dimension of x"""
            if d == 0:
                return x.min()
            return reduce(self.reducers[d](x), d-1)

        def get_confidence(cs):
            """compute the confidence"""
            cs = [reduce(c, i) for i, c in enumerate(cs)]
            conf = None
            for c in cs:
                if conf is None or conf.item() > c.item():
                    conf = c
            return conf

        if self.model_pre is not None:
            f_pre = self.model_pre(inputs)
        else:
            f_pre = inputs

        fs = [zeros(i, d) for i, d in enumerate(self.imm_dims2)]
        cs = [zeros(i, d) for i, d in enumerate(self.imm_dims2)]
        conf = torch.tensor(0, dtype=fs[0].dtype, device=fs[0].device)

        acc_fs = [zeros(i, d) for i, d in enumerate(self.imm_dims2)]
        acc_cs = [zeros(i, d) for i, d in enumerate(self.imm_dims2)]

        for i in range(max_depths):
            next_fs, next_cs = map(list, chunk(self.model(concat(f_pre, fs, cs)), 2, dim=-1))

            for i, (f, c, next_f, next_c) in enumerate(zip(fs, cs, next_fs, next_cs)):
                mask = (c > self.epsilon).float()
                next_fs[i] = f * mask + next_f * (1 - mask)
                next_cs[i] = c * mask + next_c * (1 - mask)

            # update the feature and confidence
            fs = next_fs
            last_cs, cs = cs, add(cs, next_cs)
            last_conf, conf = conf, get_confidence(cs)

            acc_fs = add(acc_fs, mul(fs, cs))
            acc_cs = add(acc_cs, cs)

            if conf > self.epsilon:
                break

        fs = div(acc_fs, acc_cs)

        if self.model_post is not None:
            f_post = self.model_pre(fs)
        else:
            f_post = fs

        return fs, torch.tensor(i, dtype=torch.float), last_cs, last_conf

    __hyperparams__ = (
        'breadth',
        'depth1', 'depth2', 'depth3',
        'input_dims', 'imm_dims1', 'imm_dims2', 'output_dims',
        'epsilon',
        'logic_model', 'logic_hidden_dim',
        'exclude_self', 'pre_residual'
    )

    __hyperparam_defaults__ = {
        'logic_model': 'mlp',
        'exclude_self': True,
        'pre_residual': False,
    }

    @classmethod
    def make_prog_block_parser(cls, parser, defaults, prefix=None):
        for k, v in cls.__hyperparam_defaults__.items():
            defaults.setdefault(k, v)

        if prefix is None:
            prefix = '--'
        else:
            prefix = '--' + str(prefix) + '-'

        parser.add_argument(prefix + 'breadth', type=int,
                            default=defaults['breadth'], metavar='N')
        parser.add_argument(prefix + 'depth1', type=int,
                            default=defaults['depth1'], metavar='N')
        parser.add_argument(prefix + 'depth2', type=int,
                            default=defaults['depth2'], metavar='N')
        parser.add_argument(prefix + 'depth3', type=int,
                            default=defaults['depth3'], metavar='N')

        parser.add_argument(prefix + 'imm-dims1', type=int,
                            default=defaults['imm_dims1'], metavar='N')
        parser.add_argument(prefix + 'imm-dims2', type=int,
                            default=defaults['imm_dims2'], metavar='N')

        parser.add_argument(prefix + 'epsilon', type=int,
                            default=defaults['epsilon'], metavar='F')

        parser.add_argument(prefix + 'logic-model',
                            default=defaults['logic_model'], choices=NeuralLogicInferenceMethod.choice_values(), metavar='T')
        parser.add_argument(prefix + 'logic-hidden-dim', type=int, nargs='+',
                            default=defaults['logic_hidden_dim'], metavar='N')

        parser.add_argument(prefix + 'exclude-self', type='bool',
                            default=defaults['exclude_self'], metavar='B')
        parser.add_argument(prefix + 'pre-residual', type='bool',
                            default=defaults['pre_residual'], metavar='B')

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
